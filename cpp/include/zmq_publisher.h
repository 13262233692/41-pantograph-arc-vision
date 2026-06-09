#ifndef ZMQ_PUBLISHER_H
#define ZMQ_PUBLISHER_H

#include <string>
#include <vector>
#include <cstdint>
#include <zmq.h>

struct ArcResultMsg {
    float x1, y1, x2, y2;
    float confidence;
    int32_t class_id;
    int64_t pts_ns;
    int32_t stream_index;
    double intensity;
};

class ZmqPublisher {
public:
    ZmqPublisher();
    ~ZmqPublisher();

    bool bind(const std::string& endpoint);
    bool connect(const std::string& endpoint);
    void close();

    int publish(const ArcResultMsg* results, int count);
    int publish_nonblock(const ArcResultMsg* results, int count);

    int receive(ArcResultMsg* out_results, int max_count, int timeout_ms);
    int receive_nonblock(ArcResultMsg* out_results, int max_count);

    bool is_valid() const { return socket_ != nullptr; }

private:
    void* context_ = nullptr;
    void* socket_ = nullptr;
    bool is_bind_ = false;
};

#endif
