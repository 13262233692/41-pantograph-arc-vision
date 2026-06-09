package result

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unsafe"

	"github.com/pebbe/zmq4"
)

type ArcSeverityLevel int32

const (
	Level1MinorSpark    ArcSeverityLevel = 1
	Level2StableArc     ArcSeverityLevel = 2
	Level3IntenseArc    ArcSeverityLevel = 3
	Level4SustainedBurn ArcSeverityLevel = 4
	Level5FatalAblation ArcSeverityLevel = 5
)

func (l ArcSeverityLevel) String() string {
	switch l {
	case Level1MinorSpark:
		return "一级飞弧"
	case Level2StableArc:
		return "二级稳定弧"
	case Level3IntenseArc:
		return "三级强弧"
	case Level4SustainedBurn:
		return "四级持续烧灼"
	case Level5FatalAblation:
		return "五级致命烧蚀"
	default:
		return fmt.Sprintf("未知(%d)", l)
	}
}

type ArcFlashResult struct {
	X1               float32
	Y1               float32
	X2               float32
	Y2               float32
	Confidence       float32
	ClassID          int32
	PTSNs            int64
	StreamIndex      int32
	Intensity        float64
	InstantEnergy    float64
	SmoothedEnergy   float64
	CumulativeEnergy float64
	SeverityLevel    ArcSeverityLevel
	TrackID          int32
	TrackFrameCount  int32
	TrackDurationNs  int64
}

type AlarmRecord struct {
	ID               int64
	Timestamp        time.Time
	StreamIndex      int32
	TrackID          int32
	SeverityLevel    ArcSeverityLevel
	BoxX1            float32
	BoxY1            float32
	BoxX2            float32
	BoxY2            float32
	Confidence       float32
	InstantEnergy    float64
	SmoothedEnergy   float64
	CumulativeEnergy float64
	TrackFrameCount  int32
	TrackDurationNs  int64
	Persistent       bool
}

type ResultReceiver struct {
	mu       sync.Mutex
	socket   *zmq4.Socket
	context  *zmq4.Context
	endpoint string
	running  bool

	results    []ArcFlashResult
	resultChan chan []ArcFlashResult
	alarmChan  chan AlarmRecord

	alarmLog *os.File
	alarms   []AlarmRecord
	alarmID  int64
}

type ReceiverConfig struct {
	Endpoint   string
	BufferSize int
	Subscribe  string
	AlarmDir   string
}

func NewResultReceiver(config ReceiverConfig) (*ResultReceiver, error) {
	ctx, err := zmq4.NewContext()
	if err != nil {
		return nil, fmt.Errorf("zmq context: %w", err)
	}

	sub, err := ctx.NewSocket(zmq4.SUB)
	if err != nil {
		ctx.Term()
		return nil, fmt.Errorf("zmq sub socket: %w", err)
	}

	if err := sub.SetSubscribe(config.Subscribe); err != nil {
		sub.Close()
		ctx.Term()
		return nil, fmt.Errorf("zmq subscribe: %w", err)
	}

	if err := sub.SetLinger(0); err != nil {
		sub.Close()
		ctx.Term()
		return nil, fmt.Errorf("zmq linger: %w", err)
	}

	hwm := 100
	if err := sub.SetRcvhwm(hwm); err != nil {
		sub.Close()
		ctx.Term()
		return nil, fmt.Errorf("zmq rcvhwm: %w", err)
	}

	bufSize := config.BufferSize
	if bufSize <= 0 {
		bufSize = 256
	}

	var alarmLog *os.File
	if config.AlarmDir != "" {
		os.MkdirAll(config.AlarmDir, 0755)
		alarmPath := filepath.Join(config.AlarmDir,
			fmt.Sprintf("alarms_%s.log", time.Now().Format("20060102_150405")))
		alarmLog, err = os.OpenFile(alarmPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			log.Printf("[ResultReceiver] Warning: failed to open alarm log: %v", err)
		}
	}

	return &ResultReceiver{
		context:    ctx,
		socket:     sub,
		endpoint:   config.Endpoint,
		resultChan: make(chan []ArcFlashResult, bufSize),
		alarmChan:  make(chan AlarmRecord, bufSize),
		alarmLog:   alarmLog,
	}, nil
}

func (rr *ResultReceiver) Connect() error {
	rr.mu.Lock()
	defer rr.mu.Unlock()

	if err := rr.socket.Connect(rr.endpoint); err != nil {
		return fmt.Errorf("zmq connect %s: %w", rr.endpoint, err)
	}

	rr.running = true
	go rr.receiveLoop()
	go rr.alarmPersistLoop()
	return nil
}

func (rr *ResultReceiver) receiveLoop() {
	poller := zmq4.NewPoller()
	poller.Add(rr.socket, zmq4.POLLIN)

	for rr.running {
		sockets, err := poller.Poll(100 * time.Millisecond)
		if err != nil {
			if rr.running {
				fmt.Printf("[ResultReceiver] poll error: %v\n", err)
			}
			continue
		}

		if len(sockets) == 0 {
			continue
		}

		msg, err := rr.socket.RecvBytes(0)
		if err != nil {
			if rr.running {
				fmt.Printf("[ResultReceiver] recv error: %v\n", err)
			}
			continue
		}

		results := rr.parseMessage(msg)
		if len(results) > 0 {
			select {
			case rr.resultChan <- results:
			default:
				fmt.Printf("[ResultReceiver] channel full, dropping %d results\n", len(results))
			}

			rr.mu.Lock()
			rr.results = append(rr.results, results...)
			if len(rr.results) > 1000 {
				rr.results = rr.results[len(rr.results)-500:]
			}
			rr.mu.Unlock()

			for _, r := range results {
				if r.SeverityLevel >= Level2StableArc {
					rr.alarmID++
					alarm := AlarmRecord{
						ID:               rr.alarmID,
						Timestamp:        time.Now(),
						StreamIndex:      r.StreamIndex,
						TrackID:          r.TrackID,
						SeverityLevel:    r.SeverityLevel,
						BoxX1:            r.X1,
						BoxY1:            r.Y1,
						BoxX2:            r.X2,
						BoxY2:            r.Y2,
						Confidence:       r.Confidence,
						InstantEnergy:    r.InstantEnergy,
						SmoothedEnergy:   r.SmoothedEnergy,
						CumulativeEnergy: r.CumulativeEnergy,
						TrackFrameCount:  r.TrackFrameCount,
						TrackDurationNs:  r.TrackDurationNs,
						Persistent:       r.SeverityLevel >= Level4SustainedBurn,
					}
					select {
					case rr.alarmChan <- alarm:
					default:
					}
				}
			}
		}
	}
}

func (rr *ResultReceiver) alarmPersistLoop() {
	for rr.running {
		select {
		case alarm, ok := <-rr.alarmChan:
			if !ok {
				return
			}
			rr.persistAlarm(alarm)

			rr.mu.Lock()
			rr.alarms = append(rr.alarms, alarm)
			if len(rr.alarms) > 500 {
				rr.alarms = rr.alarms[len(rr.alarms)-200:]
			}
			rr.mu.Unlock()
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func (rr *ResultReceiver) persistAlarm(alarm AlarmRecord) {
	durationMs := alarm.TrackDurationNs / 1e6
	line := fmt.Sprintf(
		"[%s] ID=%d Stream=%d Track=%d Severity=%s(%d) "+
			"Box=[%.1f,%.1f,%.1f,%.1f] Conf=%.3f "+
			"Energy=[instant=%.2f smoothed=%.2f cumulative=%.2f] "+
			"Frames=%d Duration=%dms Persistent=%v\n",
		alarm.Timestamp.Format("15:04:05.000"),
		alarm.ID, alarm.StreamIndex, alarm.TrackID,
		alarm.SeverityLevel, alarm.SeverityLevel,
		alarm.BoxX1, alarm.BoxY1, alarm.BoxX2, alarm.BoxY2,
		alarm.Confidence,
		alarm.InstantEnergy, alarm.SmoothedEnergy, alarm.CumulativeEnergy,
		alarm.TrackFrameCount, durationMs, alarm.Persistent,
	)

	if rr.alarmLog != nil {
		rr.alarmLog.WriteString(line)
		rr.alarmLog.Sync()
	}
}

func (rr *ResultReceiver) parseMessage(data []byte) []ArcFlashResult {
	if len(data) < 4 {
		return nil
	}

	count := int(binary.LittleEndian.Uint32(data[0:4]))
	msgSize := 4 + count*88
	if len(data) < msgSize {
		return nil
	}

	results := make([]ArcFlashResult, count)
	offset := 4

	for i := 0; i < count; i++ {
		base := offset + i*88
		results[i] = ArcFlashResult{
			X1:               float32FromBytes(data[base : base+4]),
			Y1:               float32FromBytes(data[base+4 : base+8]),
			X2:               float32FromBytes(data[base+8 : base+12]),
			Y2:               float32FromBytes(data[base+12 : base+16]),
			Confidence:       float32FromBytes(data[base+16 : base+20]),
			ClassID:          int32FromBytes(data[base+20 : base+24]),
			PTSNs:            int64FromBytes(data[base+24 : base+32]),
			StreamIndex:      int32FromBytes(data[base+32 : base+36]),
			Intensity:        float64FromBytes(data[base+36 : base+44]),
			InstantEnergy:    float64FromBytes(data[base+44 : base+52]),
			SmoothedEnergy:   float64FromBytes(data[base+52 : base+60]),
			CumulativeEnergy: float64FromBytes(data[base+60 : base+68]),
			SeverityLevel:    ArcSeverityLevel(int32FromBytes(data[base+68 : base+72])),
			TrackID:          int32FromBytes(data[base+72 : base+76]),
			TrackFrameCount:  int32FromBytes(data[base+76 : base+80]),
			TrackDurationNs:  int64FromBytes(data[base+80 : base+88]),
		}
	}

	return results
}

func (rr *ResultReceiver) Results() <-chan []ArcFlashResult {
	return rr.resultChan
}

func (rr *ResultReceiver) Alarms() <-chan AlarmRecord {
	return rr.alarmChan
}

func (rr *ResultReceiver) LatestResults() []ArcFlashResult {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	out := make([]ArcFlashResult, len(rr.results))
	copy(out, rr.results)
	return out
}

func (rr *ResultReceiver) LatestAlarms() []AlarmRecord {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	out := make([]AlarmRecord, len(rr.alarms))
	copy(out, rr.alarms)
	return out
}

func (rr *ResultReceiver) Close() {
	rr.mu.Lock()
	rr.running = false
	rr.mu.Unlock()

	if rr.alarmLog != nil {
		rr.alarmLog.Close()
		rr.alarmLog = nil
	}
	if rr.socket != nil {
		rr.socket.Close()
		rr.socket = nil
	}
	if rr.context != nil {
		rr.context.Term()
		rr.context = nil
	}
	close(rr.resultChan)
}

func float32FromBytes(b []byte) float32 {
	bits := binary.LittleEndian.Uint32(b)
	return float32frombits(bits)
}

func int32FromBytes(b []byte) int32 {
	return int32(binary.LittleEndian.Uint32(b))
}

func int64FromBytes(b []byte) int64 {
	return int64(binary.LittleEndian.Uint64(b))
}

func float64FromBytes(b []byte) float64 {
	bits := binary.LittleEndian.Uint64(b)
	return float64frombits(bits)
}

func float32frombits(b uint32) float32 {
	return *(*float32)(unsafe.Pointer(&b))
}

func float64frombits(b uint64) float64 {
	return *(*float64)(unsafe.Pointer(&b))
}
