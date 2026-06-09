package sync

import (
	"container/list"
	"sync"
	"time"
)

type TimestampedFrame struct {
	PTSNs       int64
	Width       int
	Height      int
	Channels    int
	Data        []byte
	StreamIndex int
	ReceivedAt  time.Time
}

type FrameSync struct {
	mu             sync.Mutex
	visibleFrames  *list.List
	uvFrames       *list.List
	toleranceNs    int64
	maxBufferSize  int
	onSynced       func(visible, uv *TimestampedFrame)
	dropOld        bool
}

type FrameSyncConfig struct {
	ToleranceNs   int64
	MaxBufferSize int
	DropOld       bool
}

func NewFrameSync(config FrameSyncConfig) *FrameSync {
	return &FrameSync{
		visibleFrames: list.New(),
		uvFrames:      list.New(),
		toleranceNs:   config.ToleranceNs,
		maxBufferSize: config.MaxBufferSize,
		dropOld:       config.DropOld,
	}
}

func (fs *FrameSync) SetSyncedCallback(cb func(visible, uv *TimestampedFrame)) {
	fs.onSynced = cb
}

func (fs *FrameSync) PushVisible(frame *TimestampedFrame) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.visibleFrames.PushBack(frame)
	fs.evictOld(fs.visibleFrames)
	fs.tryMatch()
}

func (fs *FrameSync) PushUV(frame *TimestampedFrame) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.uvFrames.PushBack(frame)
	fs.evictOld(fs.uvFrames)
	fs.tryMatch()
}

func (fs *FrameSync) tryMatch() {
	if fs.visibleFrames.Len() == 0 || fs.uvFrames.Len() == 0 {
		return
	}

	var bestVis *list.Element
	var bestUV *list.Element
	var bestDiff int64 = -1

	for ve := fs.visibleFrames.Front(); ve != nil; ve = ve.Next() {
		vf := ve.Value.(*TimestampedFrame)
		for ue := fs.uvFrames.Front(); ue != nil; ue = ue.Next() {
			uf := ue.Value.(*TimestampedFrame)
			diff := vf.PTSNs - uf.PTSNs
			if diff < 0 {
				diff = -diff
			}
			if bestDiff < 0 || diff < bestDiff {
				bestDiff = diff
				bestVis = ve
				bestUV = ue
			}
		}
	}

	if bestDiff < 0 || bestDiff > fs.toleranceNs {
		return
	}

	visFrame := bestVis.Value.(*TimestampedFrame)
	uvFrame := bestUV.Value.(*TimestampedFrame)

	fs.visibleFrames.Remove(bestVis)
	fs.uvFrames.Remove(bestUV)

	if fs.onSynced != nil {
		fs.onSynced(visFrame, uvFrame)
	}
}

func (fs *FrameSync) evictOld(queue *list.List) {
	for queue.Len() > fs.maxBufferSize {
		front := queue.Front()
		if front != nil {
			queue.Remove(front)
		}
	}
}

func (fs *FrameSync) Drain() {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.visibleFrames.Init()
	fs.uvFrames.Init()
}

func (fs *FrameSync) QueueLengths() (visibleLen, uvLen int) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return fs.visibleFrames.Len(), fs.uvFrames.Len()
}

type MultiStreamSync struct {
	streams    map[int]*FrameSync
	mu         sync.RWMutex
	tolerance  int64
	maxBuffer  int
	onSynced   func(streamIndex int, visible, uv *TimestampedFrame)
}

func NewMultiStreamSync(toleranceNs int64, maxBuffer int) *MultiStreamSync {
	return &MultiStreamSync{
		streams:   make(map[int]*FrameSync),
		tolerance: toleranceNs,
		maxBuffer: maxBuffer,
	}
}

func (mss *MultiStreamSync) SetSyncedCallback(cb func(streamIndex int, visible, uv *TimestampedFrame)) {
	mss.onSynced = cb
}

func (mss *MultiStreamSync) getOrCreateSync(streamIdx int) *FrameSync {
	mss.mu.Lock()
	defer mss.mu.Unlock()

	if fs, ok := mss.streams[streamIdx]; ok {
		return fs
	}

	fs := NewFrameSync(FrameSyncConfig{
		ToleranceNs:   mss.tolerance,
		MaxBufferSize: mss.maxBuffer,
		DropOld:       true,
	})

	fs.SetSyncedCallback(func(visible, uv *TimestampedFrame) {
		if mss.onSynced != nil {
			mss.onSynced(streamIdx, visible, uv)
		}
	})

	mss.streams[streamIdx] = fs
	return fs
}

func (mss *MultiStreamSync) PushVisible(streamIdx int, frame *TimestampedFrame) {
	fs := mss.getOrCreateSync(streamIdx)
	fs.PushVisible(frame)
}

func (mss *MultiStreamSync) PushUV(streamIdx int, frame *TimestampedFrame) {
	fs := mss.getOrCreateSync(streamIdx)
	fs.PushUV(frame)
}

func (mss *MultiStreamSync) Drain() {
	mss.mu.Lock()
	defer mss.mu.Unlock()

	for _, fs := range mss.streams {
		fs.Drain()
	}
}
