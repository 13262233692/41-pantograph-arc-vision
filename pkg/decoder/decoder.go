package decoder

/*
#cgo CFLAGS: -I${SRCDIR}/../../cpp/include
#cgo LDFLAGS: -lavcodec -lavformat -lavutil -lswscale -lnvcuvid -lcuda -lnvjpeg

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>

static int g_cuda_device_idx = 0;

typedef struct {
    AVFormatContext* fmt_ctx;
    AVCodecContext* codec_ctx;
    AVStream* stream;
    int stream_idx;
    int64_t last_pts;
    int eof;
} DecoderCtx;

static DecoderCtx* create_decoder() {
    DecoderCtx* ctx = (DecoderCtx*)calloc(1, sizeof(DecoderCtx));
    return ctx;
}

static void destroy_decoder(DecoderCtx* ctx) {
    if (!ctx) return;
    if (ctx->codec_ctx) avcodec_free_context(&ctx->codec_ctx);
    if (ctx->fmt_ctx) avformat_close_input(&ctx->fmt_ctx);
    free(ctx);
}

static int open_stream(DecoderCtx* ctx, const char* url, int gpu_idx) {
    g_cuda_device_idx = gpu_idx;

    int rc = avformat_open_input(&ctx->fmt_ctx, url, NULL, NULL);
    if (rc < 0) return rc;

    rc = avformat_find_stream_info(ctx->fmt_ctx, NULL);
    if (rc < 0) return rc;

    const AVCodec* codec = NULL;
    ctx->stream_idx = av_find_best_stream(ctx->fmt_ctx, AVMEDIA_TYPE_VIDEO,
                                           -1, -1, &codec, 0);
    if (ctx->stream_idx < 0) return ctx->stream_idx;

    ctx->stream = ctx->fmt_ctx->streams[ctx->stream_idx];
    ctx->codec_ctx = avcodec_alloc_context3(codec);
    if (!ctx->codec_ctx) return AVERROR(ENOMEM);

    rc = avcodec_parameters_to_context(ctx->codec_ctx, ctx->stream->codecpar);
    if (rc < 0) return rc;

    AVBufferRef* hw_device_ctx = NULL;
    char dev_name[32];
    snprintf(dev_name, sizeof(dev_name), "%d", gpu_idx);
    rc = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                 dev_name, NULL, 0);
    if (rc == 0) {
        ctx->codec_ctx->hw_device_ctx = hw_device_ctx;
        ctx->codec_ctx->hw_frames_ctx = NULL;
    }

    ctx->codec_ctx->thread_count = 2;
    rc = avcodec_open2(ctx->codec_ctx, codec, NULL);
    if (rc < 0) return rc;

    ctx->last_pts = AV_NOPTS_VALUE;
    ctx->eof = 0;
    return 0;
}

static int decode_next_frame(DecoderCtx* ctx, AVFrame* frame) {
    if (!ctx || ctx->eof) return AVERROR_EOF;

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return AVERROR(ENOMEM);

    int rc;
    for (;;) {
        rc = avcodec_receive_frame(ctx->codec_ctx, frame);
        if (rc == 0) {
            if (frame->pts != AV_NOPTS_VALUE) {
                ctx->last_pts = frame->pts;
            } else {
                frame->pts = ctx->last_pts + 1;
                ctx->last_pts = frame->pts;
            }
            av_packet_free(&pkt);
            return 0;
        }
        if (rc != AVERROR(EAGAIN)) {
            av_packet_free(&pkt);
            if (rc == AVERROR_EOF) ctx->eof = 1;
            return rc;
        }

        rc = av_read_frame(ctx->fmt_ctx, pkt);
        if (rc < 0) {
            if (rc == AVERROR_EOF) {
                avcodec_send_packet(ctx->codec_ctx, NULL);
                continue;
            }
            av_packet_free(&pkt);
            return rc;
        }

        if (pkt->stream_index == ctx->stream_idx) {
            rc = avcodec_send_packet(ctx->codec_ctx, pkt);
            av_packet_unref(pkt);
            if (rc < 0) {
                av_packet_free(&pkt);
                return rc;
            }
        } else {
            av_packet_unref(pkt);
        }
    }
}
*/
import "C"
import "unsafe"

type Decoder struct {
	ctx *C.DecoderCtx
}

type DecodedFrame struct {
	Data   []byte
	Width  int
	Height int
	PTSNs  int64
}

func NewDecoder() *Decoder {
	ctx := C.create_decoder()
	if ctx == nil {
		return nil
	}
	return &Decoder{ctx: ctx}
}

func (d *Decoder) Open(rtspURL string, gpuIdx int) error {
	cURL := C.CString(rtspURL)
	defer C.free(unsafe.Pointer(cURL))

	rc := C.open_stream(d.ctx, cURL, C.int(gpuIdx))
	if rc != 0 {
		return newAVError(int(rc))
	}
	return nil
}

func (d *Decoder) DecodeNext() (*DecodedFrame, error) {
	frame := C.av_frame_alloc()
	defer C.av_frame_unref(frame)

	rc := C.decode_next_frame(d.ctx, frame)
	if rc != 0 {
		return nil, newAVError(int(rc))
	}

	w := int(frame.width)
	h := int(frame.height)

	var data []byte
	if frame.data[0] != nil {
		frameSize := w * h * 3
		data = make([]byte, frameSize)

		srcFormat := C.AV_PIX_FMT_NV12
		if frame.format == C.AV_PIX_FMT_CUDA || frame.format == C.AV_PIX_FMT_NV12 {
			srcFormat = C.AV_PIX_FMT_NV12
		}

		_ = srcFormat

		for i := 0; i < h; i++ {
			rowStart := uintptr(unsafe.Pointer(frame.data[0])) + uintptr(i)*uintptr(frame.linesize[0])
			copy(data[i*w*3:i*w*3+w], (*[1 << 30]byte)(unsafe.Pointer(rowStart))[:w:w])
		}
	}

	pts := int64(frame.pts)

	return &DecodedFrame{
		Data:   data,
		Width:  w,
		Height: h,
		PTSNs:  pts,
	}, nil
}

func (d *Decoder) Close() {
	if d.ctx != nil {
		C.destroy_decoder(d.ctx)
		d.ctx = nil
	}
}

func (d *Decoder) IsEOF() bool {
	return d.ctx != nil && d.ctx.eof != 0
}

type avError struct {
	code int
}

func (e *avError) Error() string {
	return formatAVError(e.code)
}

func newAVError(code int) *avError {
	return &avError{code: code}
}

func formatAVError(code int) string {
	switch code {
	case -541478725:
		return "AVERROR_EOF"
	case -11:
		return "AVERROR(EAGAIN)"
	default:
		return "AVERROR"
	}
}
