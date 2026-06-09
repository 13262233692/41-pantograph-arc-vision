package result

import (
	"encoding/binary"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/pebbe/zmq4"
)

type ArcFlashResult struct {
	X1          float32
	Y1          float32
	X2          float32
	Y2          float32
	Confidence  float32
	ClassID     int32
	PTSNs       int64
	StreamIndex int32
	Intensity   float64
}

type ResultReceiver struct {
	mu       sync.Mutex
	socket   *zmq4.Socket
	context  *zmq4.Context
	endpoint string
	running  bool

	results    []ArcFlashResult
	resultChan chan []ArcFlashResult
}

type ReceiverConfig struct {
	Endpoint    string
	BufferSize  int
	Subscribe   string
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

	return &ResultReceiver{
		context:    ctx,
		socket:     sub,
		endpoint:   config.Endpoint,
		resultChan: make(chan []ArcFlashResult, bufSize),
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
		}
	}
}

func (rr *ResultReceiver) parseMessage(data []byte) []ArcFlashResult {
	if len(data) < 4 {
		return nil
	}

	count := int(binary.LittleEndian.Uint32(data[0:4]))
	msgSize := 4 + count*40
	if len(data) < msgSize {
		return nil
	}

	results := make([]ArcFlashResult, count)
	offset := 4

	for i := 0; i < count; i++ {
		base := offset + i*40
		results[i] = ArcFlashResult{
			X1:          float32FromBytes(data[base : base+4]),
			Y1:          float32FromBytes(data[base+4 : base+8]),
			X2:          float32FromBytes(data[base+8 : base+12]),
			Y2:          float32FromBytes(data[base+12 : base+16]),
			Confidence:  float32FromBytes(data[base+16 : base+20]),
			ClassID:     int32FromBytes(data[base+20 : base+24]),
			PTSNs:       int64FromBytes(data[base+24 : base+32]),
			StreamIndex: int32FromBytes(data[base+32 : base+36]),
			Intensity:   float64FromBytes(data[base+36 : base+44]),
		}
	}

	return results
}

func (rr *ResultReceiver) Results() <-chan []ArcFlashResult {
	return rr.resultChan
}

func (rr *ResultReceiver) LatestResults() []ArcFlashResult {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	out := make([]ArcFlashResult, len(rr.results))
	copy(out, rr.results)
	return out
}

func (rr *ResultReceiver) Close() {
	rr.mu.Lock()
	rr.running = false
	rr.mu.Unlock()

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
