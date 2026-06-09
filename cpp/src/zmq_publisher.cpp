#include "zmq_publisher.h"
#include <cstdio>
#include <cstring>

ZmqPublisher::ZmqPublisher() = default;

ZmqPublisher::~ZmqPublisher() {
    close();
}

bool ZmqPublisher::bind(const std::string& endpoint) {
    context_ = zmq_ctx_new();
    if (!context_) {
        fprintf(stderr, "[ZMQ] Failed to create context\n");
        return false;
    }

    socket_ = zmq_socket(context_, ZMQ_PUB);
    if (!socket_) {
        fprintf(stderr, "[ZMQ] Failed to create PUB socket\n");
        zmq_ctx_destroy(context_);
        context_ = nullptr;
        return false;
    }

    int hwm = 100;
    zmq_setsockopt(socket_, ZMQ_SNDHWM, &hwm, sizeof(hwm));

    int linger = 0;
    zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));

    if (zmq_bind(socket_, endpoint.c_str()) != 0) {
        fprintf(stderr, "[ZMQ] Failed to bind to %s: %s\n",
                endpoint.c_str(), zmq_strerror(zmq_errno()));
        zmq_close(socket_);
        zmq_ctx_destroy(context_);
        socket_ = nullptr;
        context_ = nullptr;
        return false;
    }

    is_bind_ = true;
    fprintf(stderr, "[ZMQ] PUB bound to %s\n", endpoint.c_str());
    return true;
}

bool ZmqPublisher::connect(const std::string& endpoint) {
    context_ = zmq_ctx_new();
    if (!context_) return false;

    socket_ = zmq_socket(context_, ZMQ_SUB);
    if (!socket_) {
        zmq_ctx_destroy(context_);
        context_ = nullptr;
        return false;
    }

    zmq_setsockopt(socket_, ZMQ_SUBSCRIBE, "", 0);

    int linger = 0;
    zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));

    if (zmq_connect(socket_, endpoint.c_str()) != 0) {
        zmq_close(socket_);
        zmq_ctx_destroy(context_);
        socket_ = nullptr;
        context_ = nullptr;
        return false;
    }

    is_bind_ = false;
    return true;
}

void ZmqPublisher::close() {
    if (socket_) {
        zmq_close(socket_);
        socket_ = nullptr;
    }
    if (context_) {
        zmq_ctx_destroy(context_);
        context_ = nullptr;
    }
}

int ZmqPublisher::publish(const ArcResultMsg* results, int count) {
    if (!socket_ || !results || count <= 0) return -1;

    size_t msg_size = sizeof(int32_t) + sizeof(ArcResultMsg) * count;
    std::vector<uint8_t> buffer(msg_size);

    int32_t net_count = count;
    memcpy(buffer.data(), &net_count, sizeof(int32_t));
    memcpy(buffer.data() + sizeof(int32_t), results, sizeof(ArcResultMsg) * count);

    zmq_msg_t msg;
    zmq_msg_init_size(&msg, msg_size);
    memcpy(zmq_msg_data(&msg), buffer.data(), msg_size);

    int rc = zmq_msg_send(&msg, socket_, 0);
    zmq_msg_close(&msg);

    return rc > 0 ? 0 : -1;
}

int ZmqPublisher::publish_nonblock(const ArcResultMsg* results, int count) {
    if (!socket_ || !results || count <= 0) return -1;

    size_t msg_size = sizeof(int32_t) + sizeof(ArcResultMsg) * count;
    std::vector<uint8_t> buffer(msg_size);

    int32_t net_count = count;
    memcpy(buffer.data(), &net_count, sizeof(int32_t));
    memcpy(buffer.data() + sizeof(int32_t), results, sizeof(ArcResultMsg) * count);

    zmq_msg_t msg;
    zmq_msg_init_size(&msg, msg_size);
    memcpy(zmq_msg_data(&msg), buffer.data(), msg_size);

    int rc = zmq_msg_send(&msg, socket_, ZMQ_DONTWAIT);
    zmq_msg_close(&msg);

    if (rc < 0 && zmq_errno() == EAGAIN) {
        return 1;
    }

    return rc > 0 ? 0 : -1;
}

int ZmqPublisher::receive(ArcResultMsg* out_results, int max_count, int timeout_ms) {
    if (!socket_ || !out_results) return -1;

    zmq_msg_t msg;
    zmq_msg_init(&msg);

    int rc;
    if (timeout_ms > 0) {
        zmq_pollitem_t item;
        item.socket = socket_;
        item.fd = 0;
        item.events = ZMQ_POLLIN;
        item.revents = 0;

        rc = zmq_poll(&item, 1, timeout_ms);
        if (rc <= 0) {
            zmq_msg_close(&msg);
            return rc == 0 ? 0 : -1;
        }
    }

    rc = zmq_msg_recv(&msg, socket_, 0);
    if (rc < 0) {
        zmq_msg_close(&msg);
        return -1;
    }

    size_t msg_size = zmq_msg_size(&msg);
    if (msg_size < sizeof(int32_t)) {
        zmq_msg_close(&msg);
        return -1;
    }

    const uint8_t* data = static_cast<const uint8_t*>(zmq_msg_data(&msg));
    int32_t net_count;
    memcpy(&net_count, data, sizeof(int32_t));

    int count = net_count < max_count ? net_count : max_count;
    size_t expected_size = sizeof(int32_t) + sizeof(ArcResultMsg) * net_count;

    if (msg_size >= expected_size) {
        memcpy(out_results, data + sizeof(int32_t), sizeof(ArcResultMsg) * count);
    } else {
        count = 0;
    }

    zmq_msg_close(&msg);
    return count;
}

int ZmqPublisher::receive_nonblock(ArcResultMsg* out_results, int max_count) {
    if (!socket_ || !out_results) return -1;

    zmq_msg_t msg;
    zmq_msg_init(&msg);

    int rc = zmq_msg_recv(&msg, socket_, ZMQ_DONTWAIT);
    if (rc < 0) {
        zmq_msg_close(&msg);
        if (zmq_errno() == EAGAIN) return 0;
        return -1;
    }

    size_t msg_size = zmq_msg_size(&msg);
    if (msg_size < sizeof(int32_t)) {
        zmq_msg_close(&msg);
        return -1;
    }

    const uint8_t* data = static_cast<const uint8_t*>(zmq_msg_data(&msg));
    int32_t net_count;
    memcpy(&net_count, data, sizeof(int32_t));

    int count = net_count < max_count ? net_count : max_count;
    size_t expected_size = sizeof(int32_t) + sizeof(ArcResultMsg) * net_count;

    if (msg_size >= expected_size) {
        memcpy(out_results, data + sizeof(int32_t), sizeof(ArcResultMsg) * count);
    } else {
        count = 0;
    }

    zmq_msg_close(&msg);
    return count;
}
