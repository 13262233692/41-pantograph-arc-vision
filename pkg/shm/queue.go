package shm

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp -larcvision_core -lstdc++

#include "shm_frame_queue.h"
*/
import "C"
import "unsafe"

type FrameType int32

const (
	FrameTypeVisibleLight FrameType = 0
	FrameTypeUltraviolet  FrameType = 1
)

type FrameHeader struct {
	PTSNs        int64
	Width        int32
	Height       int32
	Channels     int32
	Type         FrameType
	StreamIndex  int32
	TimestampWall int64
}

type FrameQueue struct {
	queue *C.ShmFrameQueue
	key   size_t
}

type size_t = uintptr

func CreateFrameQueue(key size_t) *FrameQueue {
	q := C.shm_queue_create(C.size_t(key))
	if q == nil {
		return nil
	}
	return &FrameQueue{queue: q, key: key}
}

func (fq *FrameQueue) Destroy() {
	if fq.queue != nil {
		C.shm_queue_destroy(fq.queue, C.size_t(fq.key))
		fq.queue = nil
	}
}

func (fq *FrameQueue) Push(header *FrameHeader, data []byte) int {
	if fq.queue == nil || header == nil || len(data) == 0 {
		return -1
	}

	var cHeader C.FrameHeader
	cHeader.pts_ns = C.int64_t(header.PTSNs)
	cHeader.width = C.int32_t(header.Width)
	cHeader.height = C.int32_t(header.Height)
	cHeader.channels = C.int32_t(header.Channels)
	cHeader.type_ = C.FrameType(header.Type)
	cHeader.stream_index = C.int32_t(header.StreamIndex)

	rc := C.shm_queue_push(
		fq.queue,
		&cHeader,
		(*C.uchar)(unsafe.Pointer(&data[0])),
	)
	return int(rc)
}

func (fq *FrameQueue) Pop() (*FrameHeader, []byte, int) {
	if fq.queue == nil {
		return nil, nil, -1
	}

	var cHeader C.FrameHeader
	outData := make([]byte, C.MAX_FRAME_SIZE)

	rc := C.shm_queue_pop(
		fq.queue,
		&cHeader,
		(*C.uchar)(unsafe.Pointer(&outData[0])),
	)

	if rc != 0 {
		return nil, nil, int(rc)
	}

	frameSize := int(cHeader.width) * int(cHeader.height) * int(cHeader.channels)
	if frameSize > int(C.MAX_FRAME_SIZE) {
		frameSize = int(C.MAX_FRAME_SIZE)
	}

	header := &FrameHeader{
		PTSNs:       int64(cHeader.pts_ns),
		Width:       int32(cHeader.width),
		Height:      int32(cHeader.height),
		Channels:    int32(cHeader.channels),
		Type:        FrameType(cHeader.type_),
		StreamIndex: int32(cHeader.stream_index),
	}

	return header, outData[:frameSize], 0
}

func (fq *FrameQueue) NativePointer() unsafe.Pointer {
	return unsafe.Pointer(fq.queue)
}
