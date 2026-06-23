#include <cuda_runtime.h>      // 引入 CUDA Runtime API，例如 cudaMalloc、cudaMemcpy、cudaDeviceSynchronize 等函数。
#include <opencv2/opencv.hpp>  // 引入 OpenCV 主头文件，用于读图、写图以及用 cv::Mat 保存图像数据。

#include <algorithm>   // 引入常用算法头文件；当前文件没有直接使用复杂算法，保留它方便后续扩展。
#include <iostream>    // 引入标准输入输出流，用于向控制台打印运行信息和错误信息。
#include <stdexcept>   // 引入标准异常类型，用 std::runtime_error 表示运行时错误。
#include <string>      // 引入 std::string，用于保存输入路径、输出路径和错误文本。

// CUDA_CHECK 是一个错误检查宏，用来包住每一次 CUDA Runtime API 调用。
// 这里使用宏而不是普通函数，是为了能用 __FILE__ 和 __LINE__ 报出真实出错位置。
// 注意：宏定义里的每一行末尾反斜杠必须保持为最后一个有效字符，否则宏会被截断。
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

template <typename T>  // 定义模板参数 T，让这个显存缓冲区可以保存 unsigned char、float 等不同类型。
class DeviceBuffer {  // DeviceBuffer 是一个 RAII 包装类，负责自动申请和释放 GPU 显存。
public:  // public 区域暴露给外部使用，例如构造、移动、取指针和查询字节数。
    DeviceBuffer() = default;  // 默认构造函数创建一个空缓冲区，此时还没有申请任何 GPU 显存。

    explicit DeviceBuffer(size_t count) {  // 带数量参数的构造函数，用于创建时立刻申请 count 个 T 类型元素。
        allocate(count);  // 调用 allocate 统一完成显存释放、计数记录和 cudaMalloc 申请。
    }

    DeviceBuffer(const DeviceBuffer&) = delete;  // 禁止拷贝构造，避免两个对象同时管理同一块 GPU 显存。
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;  // 禁止拷贝赋值，避免重复释放同一块 GPU 显存。

    DeviceBuffer(DeviceBuffer&& other) noexcept  // 移动构造函数，把 other 管理的显存所有权转移到当前对象。
        : data_(other.data_), count_(other.count_) {  // 直接接管 other 的设备指针和元素数量。
        other.data_ = nullptr;  // 将 other 的指针清空，防止 other 析构时释放已经转移出去的显存。
        other.count_ = 0;  // 将 other 的元素数量清零，使移动后的对象处于安全空状态。
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {  // 移动赋值函数，用于把另一个缓冲区的所有权转移过来。
        if (this != &other) {  // 防止自己移动给自己，否则 release 会把自己的显存提前释放掉。
            release();  // 先释放当前对象已经持有的 GPU 显存，避免内存泄漏。
            data_ = other.data_;  // 接管 other 的 GPU 指针。
            count_ = other.count_;  // 接管 other 的元素数量。
            other.data_ = nullptr;  // 清空 other 的 GPU 指针，表示它不再拥有这块显存。
            other.count_ = 0;  // 清空 other 的元素数量，维持移动后对象的安全状态。
        }
        return *this;  // 返回当前对象引用，符合 C++ 赋值运算符的常规写法。
    }

    ~DeviceBuffer() {  // 析构函数在对象生命周期结束时自动执行。
        release();  // 自动释放 GPU 显存，避免忘记调用 cudaFree 导致显存泄漏。
    }

    void allocate(size_t count) {  // allocate 用来重新申请 count 个 T 类型元素的 GPU 显存。
        release();  // 申请新显存前先释放旧显存，避免覆盖旧指针造成泄漏。
        count_ = count;  // 记录当前缓冲区包含多少个 T 类型元素。
        CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));  // 在 GPU 上申请 count_ * sizeof(T) 字节显存。
    }

    T* get() {  // 返回可写的 GPU 指针，供 cudaMemcpy 或 kernel 参数使用。
        return data_;  // 返回内部保存的设备指针；该指针指向 GPU 显存。
    }

    const T* get() const {  // const 版本的 get，允许在只读对象上获取只读设备指针。
        return data_;  // 返回内部设备指针，但调用者不能通过这个 const 指针修改数据。
    }

    size_t bytes() const {  // 返回当前缓冲区占用的总字节数。
        return count_ * sizeof(T);  // 元素数量乘以单个元素大小，就是 cudaMemcpy 需要的字节数。
    }

private:  // private 区域隐藏内部实现，外部不能直接改 data_ 和 count_。
    void release() noexcept {  // release 负责释放 GPU 显存；noexcept 表示它不会向外抛异常。
        if (data_ != nullptr) {  // 只有指针非空时才需要释放，避免对空指针做无意义操作。
            cudaFree(data_);  // 调用 CUDA Runtime 释放 GPU 显存；析构中不抛异常，所以这里不使用 CUDA_CHECK。
            data_ = nullptr;  // 释放后把指针置空，避免悬空指针被误用。
            count_ = 0;  // 释放后把元素数量清零，保持对象状态一致。
        }
    }

    T* data_ = nullptr;  // 保存 GPU 显存指针；nullptr 表示当前没有持有显存。
    size_t count_ = 0;  // 保存缓冲区里的元素数量，不是字节数。
};

__global__ void bgrToGrayKernel(const unsigned char* bgr,  // __global__ 表示这是从 CPU 发起、在 GPU 上执行的 kernel；bgr 指向 GPU 上的 BGR 输入图像。
                                unsigned char* gray,  // gray 指向 GPU 上的灰度输出图像，每个像素 1 个字节。
                                int width,  // width 是图像宽度，单位是像素。
                                int height,  // height 是图像高度，单位是像素。
                                int inputStepBytes) {  // inputStepBytes 是输入图像每一行占用的字节数，也就是 cv::Mat::step。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程负责处理的像素 x 坐标。
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程负责处理的像素 y 坐标。

    if (x >= width || y >= height) {  // 网格尺寸可能向上取整，多出来的线程需要判断是否越界。
        return;  // 越界线程不处理任何像素，直接返回。
    }

    const int bgrIndex = y * inputStepBytes + x * 3;  // 计算当前像素在 BGR 输入数组中的起始字节位置。
    const unsigned char b = bgr[bgrIndex + 0];  // 读取 B 通道；OpenCV 的彩色图默认是 BGR 顺序。
    const unsigned char g = bgr[bgrIndex + 1];  // 读取 G 通道。
    const unsigned char r = bgr[bgrIndex + 2];  // 读取 R 通道。

    const float value = 0.114f * b + 0.587f * g + 0.299f * r;  // 按常用亮度权重把 BGR 转成灰度值。
    gray[y * width + x] = static_cast<unsigned char>(__float2int_rn(value));  // 四舍五入为整数后写入灰度图对应像素。
}

__device__ unsigned char pixelAt(const unsigned char* image,  // __device__ 表示这个函数只能在 GPU 代码中调用；image 指向 GPU 上的一通道图像。
                                 int x,  // x 是要读取的像素横坐标。
                                 int y,  // y 是要读取的像素纵坐标。
                                 int width) {  // width 是图像宽度，用来把二维坐标换算成一维下标。
    return image[y * width + x];  // 按行优先布局读取 image[y][x] 对应的像素值。
}

__global__ void sobelKernel(const unsigned char* gray,  // sobelKernel 在 GPU 上执行；gray 指向灰度输入图。
                            unsigned char* edges,  // edges 指向边缘检测输出图，每个像素 1 个字节。
                            int width,  // width 是图像宽度，单位是像素。
                            int height) {  // height 是图像高度，单位是像素。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 根据 block 和 thread 编号计算当前线程的 x 坐标。
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 根据 block 和 thread 编号计算当前线程的 y 坐标。

    if (x >= width || y >= height) {  // 防止向上取整后的多余线程访问图像边界外的显存。
        return;  // 越界线程直接结束。
    }

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {  // Sobel 需要访问 3x3 邻域，最外圈像素没有完整邻域。
        edges[y * width + x] = 0;  // 这里简单把边界像素的边缘强度设为 0。
        return;  // 边界像素处理完成，直接返回。
    }

    const int tl = pixelAt(gray, x - 1, y - 1, width);  // 读取 3x3 邻域左上角像素，tl 表示 top-left。
    const int tc = pixelAt(gray, x,     y - 1, width);  // 读取 3x3 邻域上方中间像素，tc 表示 top-center。
    const int tr = pixelAt(gray, x + 1, y - 1, width);  // 读取 3x3 邻域右上角像素，tr 表示 top-right。
    const int ml = pixelAt(gray, x - 1, y,     width);  // 读取当前行左侧像素，ml 表示 middle-left。
    const int mr = pixelAt(gray, x + 1, y,     width);  // 读取当前行右侧像素，mr 表示 middle-right。
    const int bl = pixelAt(gray, x - 1, y + 1, width);  // 读取 3x3 邻域左下角像素，bl 表示 bottom-left。
    const int bc = pixelAt(gray, x,     y + 1, width);  // 读取 3x3 邻域下方中间像素，bc 表示 bottom-center。
    const int br = pixelAt(gray, x + 1, y + 1, width);  // 读取 3x3 邻域右下角像素，br 表示 bottom-right。

    const int gx = -tl + tr - 2 * ml + 2 * mr - bl + br;  // 使用 Sobel X 方向卷积核，计算水平方向灰度变化。
    const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;  // 使用 Sobel Y 方向卷积核，计算垂直方向灰度变化。
    const int absGx = gx < 0 ? -gx : gx;  // 取 gx 的绝对值，表示 X 方向边缘强度大小。
    const int absGy = gy < 0 ? -gy : gy;  // 取 gy 的绝对值，表示 Y 方向边缘强度大小。
    const int magnitude = absGx + absGy > 255 ? 255 : absGx + absGy;  // 用 absGx + absGy 近似梯度幅值，并截断到 255。

    edges[y * width + x] = static_cast<unsigned char>(magnitude);  // 把边缘强度写回输出图对应像素。
}

int main(int argc, char** argv) {  // 程序入口；argc 是命令行参数数量，argv 是命令行参数字符串数组。
    try {  // 用 try 包住主流程，让下方 catch 统一处理异常并输出错误信息。
        if (argc < 2 || argc > 4) {  // 参数数量必须是 2 到 4 个：程序名、输入图、可选边缘图、可选灰度图。
            std::cerr << "Usage: " << argv[0]  // 向标准错误输出使用说明，并显示当前可执行文件名。
                      << " <input_image> [edge_output_image] [gray_output_image]\n";  // 继续输出输入图和两个可选输出图的参数格式。
            return 1;  // 参数错误时返回非 0，表示程序执行失败。
        }

        const std::string inputPath = argv[1];  // 从第一个用户参数读取输入图片路径。
        const std::string edgeOutputPath = argc >= 3 ? argv[2] : "cuda_edges.png";  // 如果用户提供第二个参数就作为边缘图路径，否则使用默认文件名。
        const std::string grayOutputPath = argc == 4 ? argv[3] : "cuda_gray.png";  // 如果用户提供第三个参数就作为灰度图路径，否则使用默认文件名。

        cv::Mat input = cv::imread(inputPath, cv::IMREAD_COLOR);  // 用 OpenCV 从磁盘读取彩色图像，读入后格式通常是 BGR 三通道。
        if (input.empty()) {  // 如果 cv::imread 失败，OpenCV 会返回空 Mat。
            throw std::runtime_error("Failed to read image: " + inputPath);  // 抛出异常，说明图片读取失败并带上路径。
        }
        if (!input.isContinuous()) {  // 检查 Mat 数据在内存里是否连续，方便一次性拷贝到 GPU。
            input = input.clone();  // 如果不连续，就克隆出一份连续存储的数据。
        }

        const int width = input.cols;  // 从 OpenCV Mat 中取得图像宽度。
        const int height = input.rows;  // 从 OpenCV Mat 中取得图像高度。
        const size_t pixelCount = static_cast<size_t>(width) * height;  // 计算像素总数，用 size_t 避免大图乘法溢出 int。
        const size_t inputBytes = input.step * input.rows;  // 计算输入 BGR 图像总字节数，step 包含每行可能存在的对齐填充。

        DeviceBuffer<unsigned char> dInput(inputBytes);  // 在 GPU 上申请输入图缓冲区；这里按字节保存 BGR 原始数据。
        DeviceBuffer<unsigned char> dGray(pixelCount);  // 在 GPU 上申请灰度图缓冲区；每个像素 1 字节。
        DeviceBuffer<unsigned char> dEdges(pixelCount);  // 在 GPU 上申请边缘图缓冲区；每个像素 1 字节。

        CUDA_CHECK(cudaMemcpy(dInput.get(),  // 调用 cudaMemcpy，把 CPU 内存里的原图复制到 GPU 输入缓冲区。
                              input.data,  // 源地址是 OpenCV Mat 在 CPU 内存中的图像数据指针。
                              inputBytes,  // 拷贝字节数是整张输入图的实际字节数。
                              cudaMemcpyHostToDevice));  // 拷贝方向是 Host 到 Device，也就是 CPU 到 GPU。

        const dim3 block(16, 16);  // 定义每个 CUDA block 有 16x16 个线程，也就是每个 block 最多处理 256 个像素。
        const dim3 grid((width + block.x - 1) / block.x,  // 计算 x 方向需要多少个 block，向上取整覆盖整张图。
                        (height + block.y - 1) / block.y);  // 计算 y 方向需要多少个 block，向上取整覆盖整张图。

        bgrToGrayKernel<<<grid, block>>>(dInput.get(),  // 启动 BGR 转灰度 kernel，并把 GPU 输入图指针传进去。
                                         dGray.get(),  // 传入 GPU 灰度输出缓冲区指针。
                                         width,  // 传入图像宽度，供 kernel 计算边界和下标。
                                         height,  // 传入图像高度，供 kernel 计算边界。
                                         static_cast<int>(input.step));  // 传入每行字节数，帮助 kernel 正确定位 BGR 像素。
        CUDA_CHECK(cudaGetLastError());  // 检查 kernel 启动是否发生错误，例如参数非法或 launch 配置错误。
        CUDA_CHECK(cudaDeviceSynchronize());  // 等待 GPU 完成灰度转换，并检查 kernel 执行过程中的异步错误。

        cv::Mat gray(height, width, CV_8UC1);  // 在 CPU 内存中创建一张 8 位单通道灰度图，用于接收 GPU 结果。
        CUDA_CHECK(cudaMemcpy(gray.data,  // 把 GPU 灰度结果复制回 CPU Mat 的数据区。
                              dGray.get(),  // 源地址是 GPU 上的灰度缓冲区。
                              dGray.bytes(),  // 拷贝字节数等于灰度缓冲区总字节数。
                              cudaMemcpyDeviceToHost));  // 拷贝方向是 Device 到 Host，也就是 GPU 到 CPU。

        if (!cv::imwrite(grayOutputPath, gray)) {  // 用 OpenCV 把灰度图写到磁盘，并检查是否写入成功。
            throw std::runtime_error("Failed to write image: " + grayOutputPath);  // 写入失败时抛出异常并带上目标路径。
        }

        sobelKernel<<<grid, block>>>(dGray.get(),  // 启动 Sobel 边缘检测 kernel，输入是 GPU 上的灰度图。
                                     dEdges.get(),  // 输出是 GPU 上的边缘图缓冲区。
                                     width,  // 传入图像宽度，供 kernel 判断边界和计算下标。
                                     height);  // 传入图像高度，供 kernel 判断边界。
        CUDA_CHECK(cudaGetLastError());  // 检查 Sobel kernel 启动是否成功。
        CUDA_CHECK(cudaDeviceSynchronize());  // 等待 Sobel kernel 执行完成，并捕获异步执行错误。

        cv::Mat edges(height, width, CV_8UC1);  // 在 CPU 内存中创建一张 8 位单通道边缘图。
        CUDA_CHECK(cudaMemcpy(edges.data,  // 把 GPU 上的 Sobel 输出复制到 CPU Mat 数据区。
                              dEdges.get(),  // 源地址是 GPU 上的边缘图缓冲区。
                              dEdges.bytes(),  // 拷贝字节数等于边缘图缓冲区大小。
                              cudaMemcpyDeviceToHost));  // 拷贝方向是 GPU 到 CPU。

        if (!cv::imwrite(edgeOutputPath, edges)) {  // 用 OpenCV 把边缘图写到磁盘，并判断写入是否成功。
            throw std::runtime_error("Failed to write image: " + edgeOutputPath);  // 写入失败时抛出异常并带上目标路径。
        }

        std::cout << "Input:  " << inputPath << '\n'  // 向控制台输出输入图路径。
                  << "Gray:   " << grayOutputPath << '\n'  // 向控制台输出灰度图保存路径。
                  << "Edges:  " << edgeOutputPath << '\n'  // 向控制台输出边缘图保存路径。
                  << "Size:   " << width << " x " << height << '\n';  // 向控制台输出图像尺寸。
        return 0;  // 主流程成功完成，返回 0 表示程序正常结束。
    } catch (const std::exception& ex) {  // 捕获标准异常，包括 runtime_error 和其他 std::exception 子类。
        std::cerr << ex.what() << '\n';  // 把异常消息输出到标准错误，方便定位失败原因。
        return 1;  // 出现异常时返回非 0，表示程序执行失败。
    }
}
