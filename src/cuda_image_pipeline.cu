#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(status) +              \
                                     " at " + __FILE__ + ":" +               \
                                     std::to_string(__LINE__));                \
        }                                                                      \
    } while (false)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t count) {
        allocate(count);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        release();
    }

    void allocate(size_t count) {
        release();
        count_ = count;
        CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
    }

    T* get() {
        return data_;
    }

    const T* get() const {
        return data_;
    }

    size_t bytes() const {
        return count_ * sizeof(T);
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    T* data_ = nullptr;
    size_t count_ = 0;
};

__global__ void bgrToGrayKernel(const unsigned char* bgr,
                                unsigned char* gray,
                                int width,
                                int height,
                                int inputStepBytes) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int bgrIndex = y * inputStepBytes + x * 3;
    const unsigned char b = bgr[bgrIndex + 0];
    const unsigned char g = bgr[bgrIndex + 1];
    const unsigned char r = bgr[bgrIndex + 2];

    const float value = 0.114f * b + 0.587f * g + 0.299f * r;
    gray[y * width + x] = static_cast<unsigned char>(__float2int_rn(value));
}

__device__ unsigned char pixelAt(const unsigned char* image,
                                 int x,
                                 int y,
                                 int width) {
    return image[y * width + x];
}

__global__ void sobelKernel(const unsigned char* gray,
                            unsigned char* edges,
                            int width,
                            int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        edges[y * width + x] = 0;
        return;
    }

    const int tl = pixelAt(gray, x - 1, y - 1, width);
    const int tc = pixelAt(gray, x,     y - 1, width);
    const int tr = pixelAt(gray, x + 1, y - 1, width);
    const int ml = pixelAt(gray, x - 1, y,     width);
    const int mr = pixelAt(gray, x + 1, y,     width);
    const int bl = pixelAt(gray, x - 1, y + 1, width);
    const int bc = pixelAt(gray, x,     y + 1, width);
    const int br = pixelAt(gray, x + 1, y + 1, width);

    const int gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
    const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
    const int absGx = gx < 0 ? -gx : gx;
    const int absGy = gy < 0 ? -gy : gy;
    const int magnitude = absGx + absGy > 255 ? 255 : absGx + absGy;

    edges[y * width + x] = static_cast<unsigned char>(magnitude);
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <input_image> [edge_output_image] [gray_output_image]\n";
            return 1;
        }

        const std::string inputPath = argv[1];
        const std::string edgeOutputPath = argc >= 3 ? argv[2] : "cuda_edges.png";
        const std::string grayOutputPath = argc == 4 ? argv[3] : "cuda_gray.png";

        cv::Mat input = cv::imread(inputPath, cv::IMREAD_COLOR);
        if (input.empty()) {
            throw std::runtime_error("Failed to read image: " + inputPath);
        }
        if (!input.isContinuous()) {
            input = input.clone();
        }

        const int width = input.cols;
        const int height = input.rows;
        const size_t pixelCount = static_cast<size_t>(width) * height;
        const size_t inputBytes = input.step * input.rows;

        DeviceBuffer<unsigned char> dInput(inputBytes);
        DeviceBuffer<unsigned char> dGray(pixelCount);
        DeviceBuffer<unsigned char> dEdges(pixelCount);

        CUDA_CHECK(cudaMemcpy(dInput.get(),
                              input.data,
                              inputBytes,
                              cudaMemcpyHostToDevice));

        const dim3 block(16, 16);
        const dim3 grid((width + block.x - 1) / block.x,
                        (height + block.y - 1) / block.y);

        bgrToGrayKernel<<<grid, block>>>(dInput.get(),
                                         dGray.get(),
                                         width,
                                         height,
                                         static_cast<int>(input.step));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cv::Mat gray(height, width, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(gray.data,
                              dGray.get(),
                              dGray.bytes(),
                              cudaMemcpyDeviceToHost));

        if (!cv::imwrite(grayOutputPath, gray)) {
            throw std::runtime_error("Failed to write image: " + grayOutputPath);
        }

        sobelKernel<<<grid, block>>>(dGray.get(),
                                     dEdges.get(),
                                     width,
                                     height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cv::Mat edges(height, width, CV_8UC1);
        CUDA_CHECK(cudaMemcpy(edges.data,
                              dEdges.get(),
                              dEdges.bytes(),
                              cudaMemcpyDeviceToHost));

        if (!cv::imwrite(edgeOutputPath, edges)) {
            throw std::runtime_error("Failed to write image: " + edgeOutputPath);
        }

        std::cout << "Input:  " << inputPath << '\n'
                  << "Gray:   " << grayOutputPath << '\n'
                  << "Edges:  " << edgeOutputPath << '\n'
                  << "Size:   " << width << " x " << height << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
