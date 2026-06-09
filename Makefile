.PHONY: build-cpp build-go build clean

BUILD_DIR ?= build
TRT_ROOT ?= /usr/local/tensorrt
CGO_CFLAGS ?=
CGO_LDFLAGS ?=

build-cpp:
	cmake -B $(BUILD_DIR) \
		-DTRT_ROOT=$(TRT_ROOT) \
		-DCMAKE_BUILD_TYPE=Release \
		-S .
	cmake --build $(BUILD_DIR) -j$$(nproc 2>/dev/null || echo 4)

build-go: build-cpp
	export CGO_CFLAGS="-I$$(pwd)/cpp/include $(CGO_CFLAGS)" && \
	export CGO_LDFLAGS="-L$$(pwd)/$(BUILD_DIR)/cpp -larcvision_core -lstdc++ -lzmq -lcuda -lcudart -lnvinfer $(CGO_LDFLAGS)" && \
	go build -o $(BUILD_DIR)/arcvision ./cmd/arcvision

build: build-go

run: build
	LD_LIBRARY_PATH=$$(pwd)/$(BUILD_DIR)/cpp:$$LD_LIBRARY_PATH \
	$(BUILD_DIR)/arcvision

clean:
	rm -rf $(BUILD_DIR)
	go clean
