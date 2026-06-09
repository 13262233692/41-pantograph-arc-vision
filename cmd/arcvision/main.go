package main

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/cpp -larcvision_core -lstdc++ -lzmq -lcuda -lcudart -lnvinfer

#include "shm_frame_queue.h"

extern int arcvision_engine_init(const char* visible_engine_path,
                                 const char* uv_engine_path,
                                 int device_id,
                                 float visible_thresh,
                                 float uv_thresh,
                                 float nms_thresh,
                                 const char* zmq_endpoint,
                                 int num_streams);

extern void arcvision_engine_set_stream_queues(int stream_index,
                                                void* visible_queue,
                                                void* uv_queue);

extern int arcvision_engine_process_frame(const unsigned char* visible_gpu,
                                          const unsigned char* uv_gpu,
                                          int width, int height,
                                          int64_t pts_ns,
                                          int32_t stream_index);

extern void arcvision_engine_run_loop(int64_t sync_tolerance_ns);

extern void arcvision_engine_shutdown();
*/
import "C"
import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
	"unsafe"

	"arc-vision/pkg/decoder"
	"arc-vision/pkg/result"
	"arc-vision/pkg/shm"
	"arc-vision/pkg/sync"
)

type StreamConfig struct {
	VisibleRTSP string `yaml:"visible_rtsp"`
	UVRTSP      string `yaml:"uv_rtsp"`
	StreamIndex int   `yaml:"stream_index"`
}

type AppConfig struct {
	GPUDeviceID      int            `yaml:"gpu_device_id"`
	VisibleEngine    string         `yaml:"visible_engine_path"`
	UVEngine         string         `yaml:"uv_engine_path"`
	VisibleThreshold float32        `yaml:"visible_score_threshold"`
	UVThreshold      float32        `yaml:"uv_score_threshold"`
	NMSThreshold     float32        `yaml:"nms_threshold"`
	SyncToleranceMs  float64        `yaml:"sync_tolerance_ms"`
	ZmqEndpoint      string         `yaml:"zmq_endpoint"`
	Streams          []StreamConfig `yaml:"streams"`
	ShmKeyBase       uintptr        `yaml:"shm_key_base"`
}

func loadConfig(path string) *AppConfig {
	cfg := &AppConfig{
		GPUDeviceID:      0,
		VisibleEngine:    "models/pantograph_detect.trt",
		UVEngine:         "models/arc_flash_detect.trt",
		VisibleThreshold: 0.5,
		UVThreshold:      0.4,
		NMSThreshold:     0.45,
		SyncToleranceMs:  5.0,
		ZmqEndpoint:      "ipc:///tmp/arcvision_results",
		ShmKeyBase:       0x41524300,
		Streams: []StreamConfig{
			{
				VisibleRTSP: "rtsp://192.168.1.100:554/visible",
				UVRTSP:      "rtsp://192.168.1.101:554/uv",
				StreamIndex: 0,
			},
		},
	}
	return cfg
}

func main() {
	cfg := loadConfig("configs/config.yaml")
	numStreams := len(cfg.Streams)

	log.Println("[ArcVision] Starting pantograph arc flash detection system")
	log.Printf("[ArcVision] GPU Device: %d", cfg.GPUDeviceID)
	log.Printf("[ArcVision] Visible engine: %s", cfg.VisibleEngine)
	log.Printf("[ArcVision] UV engine: %s", cfg.UVEngine)
	log.Printf("[ArcVision] ZMQ endpoint: %s", cfg.ZmqEndpoint)
	log.Printf("[ArcVision] Number of streams: %d", numStreams)

	streamQueues := make(map[int]*shm.FrameQueue, numStreams)
	for _, sc := range cfg.Streams {
		visKey := cfg.ShmKeyBase + uintptr(sc.StreamIndex*2)
		uvKey := cfg.ShmKeyBase + uintptr(sc.StreamIndex*2+1)

		visQ := shm.CreateFrameQueue(visKey)
		uvQ := shm.CreateFrameQueue(uvKey)
		if visQ == nil || uvQ == nil {
			log.Fatalf("[ArcVision] Failed to create shared memory queues for stream %d", sc.StreamIndex)
		}
		defer visQ.Destroy()
		defer uvQ.Destroy()

		streamQueues[sc.StreamIndex] = visQ
		_ = uvQ
	}

	cVisibleEngine := C.CString(cfg.VisibleEngine)
	cUVEngine := C.CString(cfg.UVEngine)
	cZmqEndpoint := C.CString(cfg.ZmqEndpoint)
	defer C.free(unsafe.Pointer(cVisibleEngine))
	defer C.free(unsafe.Pointer(cUVEngine))
	defer C.free(unsafe.Pointer(cZmqEndpoint))

	rc := C.arcvision_engine_init(
		cVisibleEngine,
		cUVEngine,
		C.int(cfg.GPUDeviceID),
		C.float(cfg.VisibleThreshold),
		C.float(cfg.UVThreshold),
		C.float(cfg.NMSThreshold),
		cZmqEndpoint,
		C.int(numStreams),
	)
	if rc != 0 {
		log.Fatalf("[ArcVision] Engine init failed: %d", rc)
	}
	defer C.arcvision_engine_shutdown()

	for _, sc := range cfg.Streams {
		visKey := cfg.ShmKeyBase + uintptr(sc.StreamIndex*2)
		uvKey := cfg.ShmKeyBase + uintptr(sc.StreamIndex*2+1)

		visQ := shm.CreateFrameQueue(visKey)
		uvQ := shm.CreateFrameQueue(uvKey)

		C.arcvision_engine_set_stream_queues(
			C.int(sc.StreamIndex),
			visQ.NativePointer(),
			uvQ.NativePointer(),
		)
	}

	syncToleranceNs := int64(cfg.SyncToleranceMs * float64(time.Millisecond))
	multiSync := sync.NewMultiStreamSync(syncToleranceNs, 30)

	resultReceiver, err := result.NewResultReceiver(result.ReceiverConfig{
		Endpoint:   cfg.ZmqEndpoint,
		BufferSize: 512,
		Subscribe:  "",
		AlarmDir:   "alarms",
	})
	if err != nil {
		log.Fatalf("[ArcVision] Failed to create result receiver: %v", err)
	}
	defer resultReceiver.Close()

	if err := resultReceiver.Connect(); err != nil {
		log.Fatalf("[ArcVision] Failed to connect result receiver: %v", err)
	}

	multiSync.SetSyncedCallback(func(streamIdx int, vis, uv *sync.TimestampedFrame) {
		visQ := streamQueues[streamIdx]
		if visQ == nil {
			return
		}
		uvKey := cfg.ShmKeyBase + uintptr(streamIdx*2+1)
		uvQ := shm.CreateFrameQueue(uvKey)
		if uvQ == nil {
			return
		}

		visHeader := &shm.FrameHeader{
			PTSNs:       vis.PTSNs,
			Width:       int32(vis.Width),
			Height:      int32(vis.Height),
			Channels:    int32(vis.Channels),
			Type:        shm.FrameTypeVisibleLight,
			StreamIndex: int32(streamIdx),
		}
		visQ.Push(visHeader, vis.Data)

		uvHeader := &shm.FrameHeader{
			PTSNs:       uv.PTSNs,
			Width:       int32(uv.Width),
			Height:      int32(uv.Height),
			Channels:    int32(uv.Channels),
			Type:        shm.FrameTypeUltraviolet,
			StreamIndex: int32(streamIdx),
		}
		uvQ.Push(uvHeader, uv.Data)
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go C.arcvision_engine_run_loop(C.int64_t(syncToleranceNs))

	for _, streamCfg := range cfg.Streams {
		go ingestStream(ctx, streamCfg, cfg.GPUDeviceID, multiSync)
	}

	go resultAggregator(ctx, resultReceiver)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	<-sigCh
	log.Println("[ArcVision] Shutting down...")
	cancel()

	time.Sleep(500 * time.Millisecond)
	log.Println("[ArcVision] Stopped")
}

func ingestStream(ctx context.Context, cfg StreamConfig, gpuIdx int, mss *sync.MultiStreamSync) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		visDec := decoder.NewDecoder()
		if visDec == nil {
			log.Printf("[Stream %d] Failed to create visible decoder", cfg.StreamIndex)
			time.Sleep(2 * time.Second)
			continue
		}

		if err := visDec.Open(cfg.VisibleRTSP, gpuIdx); err != nil {
			log.Printf("[Stream %d] Failed to open visible RTSP %s: %v",
				cfg.StreamIndex, cfg.VisibleRTSP, err)
			visDec.Close()
			time.Sleep(2 * time.Second)
			continue
		}

		uvDec := decoder.NewDecoder()
		if uvDec == nil {
			log.Printf("[Stream %d] Failed to create UV decoder", cfg.StreamIndex)
			visDec.Close()
			time.Sleep(2 * time.Second)
			continue
		}

		if err := uvDec.Open(cfg.UVRTSP, gpuIdx); err != nil {
			log.Printf("[Stream %d] Failed to open UV RTSP %s: %v",
				cfg.StreamIndex, cfg.UVRTSP, err)
			visDec.Close()
			uvDec.Close()
			time.Sleep(2 * time.Second)
			continue
		}

		log.Printf("[Stream %d] Connected to RTSP streams", cfg.StreamIndex)

		visDone := make(chan struct{})
		uvDone := make(chan struct{})

		go func() {
			defer close(visDone)
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}

				frame, err := visDec.DecodeNext()
				if err != nil {
					log.Printf("[Stream %d] Visible decode error: %v", cfg.StreamIndex, err)
					return
				}

				mss.PushVisible(cfg.StreamIndex, &sync.TimestampedFrame{
					PTSNs:       frame.PTSNs,
					Width:       frame.Width,
					Height:      frame.Height,
					Channels:    3,
					Data:        frame.Data,
					StreamIndex: cfg.StreamIndex,
					ReceivedAt:  time.Now(),
				})
			}
		}()

		go func() {
			defer close(uvDone)
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}

				frame, err := uvDec.DecodeNext()
				if err != nil {
					log.Printf("[Stream %d] UV decode error: %v", cfg.StreamIndex, err)
					return
				}

				mss.PushUV(cfg.StreamIndex, &sync.TimestampedFrame{
					PTSNs:       frame.PTSNs,
					Width:       frame.Width,
					Height:      frame.Height,
					Channels:    1,
					Data:        frame.Data,
					StreamIndex: cfg.StreamIndex,
					ReceivedAt:  time.Now(),
				})
			}
		}()

		<-visDone
		<-uvDone

		visDec.Close()
		uvDec.Close()

		log.Printf("[Stream %d] Stream disconnected, reconnecting in 2s...", cfg.StreamIndex)
		time.Sleep(2 * time.Second)
	}
}

func resultAggregator(ctx context.Context, receiver *result.ResultReceiver) {
	for {
		select {
		case <-ctx.Done():
			return
		case results, ok := <-receiver.Results():
			if !ok {
				return
			}
			for _, r := range results {
				if r.SeverityLevel >= result.Level2StableArc {
					durationMs := r.TrackDurationNs / 1e6
					log.Printf("[ALERT] %s! Stream=%d Track=%d PTS=%d "+
						"Box=[%.1f,%.1f,%.1f,%.1f] Conf=%.3f "+
						"Energy=[instant=%.2f smoothed=%.2f cumulative=%.2f] "+
						"Frames=%d Duration=%dms",
						r.SeverityLevel, r.StreamIndex, r.TrackID, r.PTSNs,
						r.X1, r.Y1, r.X2, r.Y2, r.Confidence,
						r.InstantEnergy, r.SmoothedEnergy, r.CumulativeEnergy,
						r.TrackFrameCount, durationMs)

					if r.SeverityLevel >= result.Level4SustainedBurn {
						log.Printf("[CRITICAL] ⚠ 致命烧蚀风险! Stream=%d Track=%d "+
							"累计能量=%.2f 持续时间=%dms 立即断电!",
							r.StreamIndex, r.TrackID,
							r.CumulativeEnergy, durationMs)
					}
				}
			}
		}
	}
}
