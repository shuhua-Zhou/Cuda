#include <cuda_runtime.h>  // 引入 CUDA Runtime API，提供 cudaMalloc、cudaMemcpy、cudaFree、cudaDeviceSynchronize 等基础函数。

#include <opencv2/core/cuda.hpp>     // 引入 OpenCV CUDA 基础设施，例如 cv::cuda::GpuMat 和 CUDA 设备查询函数。
#include <opencv2/cudaarithm.hpp>    // 引入 OpenCV CUDA 算术模块，例如 cv::cuda::abs 和 cv::cuda::add。
#include <opencv2/cudafilters.hpp>   // 引入 OpenCV CUDA 滤波模块，例如 cv::cuda::createSobelFilter。
#include <opencv2/cudaimgproc.hpp>   // 引入 OpenCV CUDA 图像处理模块，例如 cv::cuda::cvtColor。
#include <opencv2/opencv.hpp>        // 引入 OpenCV 常用主头文件，用于 cv::Mat、imread、imwrite 等 CPU 侧功能。

#include <algorithm>   // 引入 std::transform，用于把模式字符串统一转换成小写。
#include <cctype>      // 引入 std::tolower，用于字符级大小写转换。
#include <iostream>    // 引入 std::cout 和 std::cerr，用于输出运行信息和错误信息。
#include <stdexcept>   // 引入 std::runtime_error，用于用异常报告运行时错误。
#include <string>      // 引入 std::string，用于保存路径、模式名和错误文本。

// 自写 CUDA kernel 时，CUDA Runtime API 的错误通常不会自动变成 C++ 异常。
// 这个宏把 cudaMalloc、cudaMemcpy、cudaDeviceSynchronize 等调用统一包起来。
// 注意：下面宏定义中的每个反斜杠必须保持为该行最后一个有效字符，所以不在宏行尾追加中文注释。
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

template <typename T>  // 定义模板参数 T，让同一个显存缓冲区类可以保存 unsigned char、float、int 等不同元素类型。
class DeviceBuffer {  // 定义一个 RAII 类，用 C++ 对象生命周期自动管理 GPU 显存。
public:  // public 区域提供外部可调用的构造、移动、申请、取指针和查询大小接口。
    DeviceBuffer() = default;  // 默认构造函数创建一个空对象，此时 data_ 为 nullptr，不持有任何 GPU 显存。

    explicit DeviceBuffer(size_t count) {  // 带元素数量的构造函数，用于创建对象时立刻申请 GPU 显存。
        allocate(count);  // 复用 allocate 函数完成显存申请，避免构造函数和普通申请逻辑重复。
    }  // 结束带参构造函数。

    DeviceBuffer(const DeviceBuffer&) = delete;  // 禁止拷贝构造，避免两个对象同时认为自己拥有同一块 GPU 显存。
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;  // 禁止拷贝赋值，避免赋值后出现重复释放同一块显存的问题。

    DeviceBuffer(DeviceBuffer&& other) noexcept  // 定义移动构造函数，用于把 other 的显存所有权转移到当前对象。
        : data_(other.data_), count_(other.count_) {  // 用初始化列表接管 other 的设备指针和元素数量。
        other.data_ = nullptr;  // 把 other 的设备指针清空，避免 other 析构时释放已经转移走的显存。
        other.count_ = 0;  // 把 other 的元素数量清零，使移动后的对象保持安全的空状态。
    }  // 结束移动构造函数。

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {  // 定义移动赋值函数，用于把另一个缓冲区的所有权转移给当前对象。
        if (this != &other) {  // 判断是否不是自我移动赋值，防止自己把自己的显存先释放掉。
            release();  // 当前对象如果已经持有显存，先释放旧显存，避免内存泄漏。
            data_ = other.data_;  // 接管 other 的 GPU 显存指针。
            count_ = other.count_;  // 接管 other 的元素数量。
            other.data_ = nullptr;  // 清空 other 的设备指针，表示 other 不再拥有那块显存。
            other.count_ = 0;  // 清空 other 的元素数量，使 other 回到空缓冲区状态。
        }  // 结束自我赋值保护分支。
        return *this;  // 返回当前对象引用，符合 C++ 赋值运算符约定。
    }  // 结束移动赋值函数。

    ~DeviceBuffer() {  // 析构函数会在对象离开作用域时自动执行。
        release();  // 自动释放 GPU 显存，避免调用者忘记 cudaFree。
    }  // 结束析构函数。

    void allocate(size_t count) {  // 申请 count 个 T 类型元素大小的 GPU 显存。
        release();  // 申请新显存前释放旧显存，避免覆盖 data_ 导致旧显存泄漏。
        count_ = count;  // 保存元素数量，后续 bytes() 会用它计算总字节数。
        CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));  // 调用 CUDA 在 GPU 上申请 count_ * sizeof(T) 字节。
    }  // 结束 allocate 函数。

    T* get() {  // 返回可写设备指针，供 cudaMemcpy 或 CUDA kernel 参数使用。
        return data_;  // 返回内部保存的 GPU 显存地址。
    }  // 结束非 const get 函数。

    const T* get() const {  // 返回只读设备指针，允许 const DeviceBuffer 对象暴露只读访问。
        return data_;  // 返回内部保存的 GPU 显存地址，但调用者不能通过该 const 指针修改数据。
    }  // 结束 const get 函数。

    size_t bytes() const {  // 查询当前缓冲区占用的总字节数。
        return count_ * sizeof(T);  // 元素数量乘以单个元素大小，就是 cudaMemcpy 需要的字节数。
    }  // 结束 bytes 函数。

private:  // private 区域隐藏内部资源管理细节，外部不能直接修改 data_ 和 count_。
    void release() noexcept {  // 释放当前对象持有的 GPU 显存；noexcept 表示析构路径不会抛异常。
        if (data_ != nullptr) {  // 只有当前确实持有显存时才调用 cudaFree。
            cudaFree(data_);  // 释放 GPU 显存；这里不使用 CUDA_CHECK，因为析构函数中不应抛异常。
            data_ = nullptr;  // 释放后立刻置空，避免留下悬空指针。
            count_ = 0;  // 释放后把元素数量清零，保持对象状态一致。
        }  // 结束非空指针判断。
    }  // 结束 release 函数。

    T* data_ = nullptr;  // 保存 GPU 显存指针；nullptr 表示当前没有持有显存。
    size_t count_ = 0;  // 保存缓冲区中的元素数量；注意它不是字节数。
};  // 结束 DeviceBuffer 类定义。

enum class PipelineMode {  // 定义程序运行模式，明确区分自写 CUDA、OpenCV CUDA 和双路径对比。
    CustomCuda,  // 只运行自己写的 CUDA kernel 版本。
    OpenCvCuda,  // 只运行 OpenCV CUDA 模块版本。
    OpenCvCustomCuda,  // 使用 OpenCV GpuMat 上传下载，但中间仍然运行自写 CUDA kernel。
    Both  // 同时运行两种版本，方便对比输出结果。
};  // 结束 PipelineMode 枚举。

std::string toLower(std::string text) {  // 把字符串转换成小写，便于命令行模式参数大小写不敏感。
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {  // 遍历字符串中的每个字符并原地替换成小写。
        return static_cast<char>(std::tolower(ch));  // std::tolower 返回 int，这里转回 char 保存到 string。
    });  // 结束 std::transform 调用。
    return text;  // 返回转换后的小写字符串。
}  // 结束 toLower 函数。

bool isModeText(const std::string& text) {  // 判断一个字符串是不是合法的模式参数。
    const std::string value = toLower(text);  // 先转成小写，避免 Custom、CUSTOM 等写法不匹配。
    return value == "custom" || value == "opencv-cuda" ||
           value == "opencv-custom" || value == "both";  // 任意合法模式匹配就返回 true。
}  // 结束 isModeText 函数。

PipelineMode parseMode(const std::string& text) {  // 把命令行字符串解析成 PipelineMode 枚举。
    const std::string value = toLower(text);  // 统一转成小写，简化后面的比较逻辑。
    if (value == "custom") {  // 如果用户选择 custom 模式。
        return PipelineMode::CustomCuda;  // 返回自写 CUDA kernel 模式。
    }  // 结束 custom 判断。
    if (value == "opencv-cuda") {  // 如果用户选择 opencv-cuda 模式。
        return PipelineMode::OpenCvCuda;  // 返回 OpenCV CUDA 模式。
    }  // 结束 opencv-cuda 判断。
    if (value == "opencv-custom") {  // 如果用户选择 opencv-custom 模式。
        return PipelineMode::OpenCvCustomCuda;  // 返回 OpenCV GpuMat 加自写 CUDA kernel 的混合模式。
    }  // 结束 opencv-custom 判断。
    if (value == "both") {  // 如果用户选择 both 模式。
        return PipelineMode::Both;  // 返回双路径同时运行模式。
    }  // 结束 both 判断。
    throw std::runtime_error("Unknown mode: " + text);  // 未匹配任何合法模式时抛出异常。
}  // 结束 parseMode 函数。

const char* modeName(PipelineMode mode) {  // 把 PipelineMode 枚举转换成用于打印的字符串。
    switch (mode) {  // 根据枚举值分支返回对应文本。
    case PipelineMode::CustomCuda:  // 当前模式是自写 CUDA kernel。
        return "custom";  // 返回 custom 文本。
    case PipelineMode::OpenCvCuda:  // 当前模式是 OpenCV CUDA。
        return "opencv-cuda";  // 返回 opencv-cuda 文本。
    case PipelineMode::OpenCvCustomCuda:  // 当前模式是 OpenCV GpuMat 加自写 CUDA kernel。
        return "opencv-custom";  // 返回 opencv-custom 文本。
    case PipelineMode::Both:  // 当前模式是两种实现都运行。
        return "both";  // 返回 both 文本。
    }  // 结束 switch。
    return "unknown";  // 理论上不会走到这里，保留兜底返回值避免编译器警告。
}  // 结束 modeName 函数。

std::string addSuffixBeforeExtension(const std::string& path,  // 定义给文件路径追加后缀的工具函数。
                                     const std::string& suffix) {  // suffix 是要插入到扩展名前面的文本。
    const size_t slash = path.find_last_of("\\/");  // 查找最后一个路径分隔符，用于判断点号是否属于文件名。
    const size_t dot = path.find_last_of('.');  // 查找最后一个点号，用于识别扩展名位置。
    if (dot != std::string::npos &&  // 确认路径中存在点号。
        (slash == std::string::npos || dot > slash)) {  // 确认点号位于最后一个路径分隔符之后，也就是属于文件名。
        return path.substr(0, dot) + suffix + path.substr(dot);  // 在扩展名前插入后缀，例如 a.png 变成 a_suffix.png。
    }  // 结束有扩展名路径的处理。
    return path + suffix;  // 如果没有扩展名，就直接把后缀追加到路径末尾。
}  // 结束 addSuffixBeforeExtension 函数。

void zeroImageBorder(cv::Mat& image) {  // 把图像最外圈像素置零，用于让 OpenCV Sobel 输出边界更接近自写 kernel 的处理。
    if (image.empty()) {  // 如果传入的是空图像，就没有任何像素需要处理。
        return;  // 直接返回，避免访问不存在的行列。
    }  // 结束空图判断。
    image.row(0).setTo(cv::Scalar::all(0));  // 把第一行所有像素置为 0。
    image.row(image.rows - 1).setTo(cv::Scalar::all(0));  // 把最后一行所有像素置为 0。
    image.col(0).setTo(cv::Scalar::all(0));  // 把第一列所有像素置为 0。
    image.col(image.cols - 1).setTo(cv::Scalar::all(0));  // 把最后一列所有像素置为 0。
}  // 结束 zeroImageBorder 函数。

void writeImageOrThrow(const std::string& path, const cv::Mat& image) {  // 写图工具函数，失败时用异常报告路径。
    if (!cv::imwrite(path, image)) {  // 调用 OpenCV 把 Mat 写入磁盘，并判断是否失败。
        throw std::runtime_error("Failed to write image: " + path);  // 写入失败时抛出异常，带上目标文件路径。
    }  // 结束写入失败判断。
}  // 结束 writeImageOrThrow 函数。

// 这是自写 CUDA kernel 版本的 BGR 转灰度函数。
// 每个 GPU 线程负责一个像素，手动从 BGR 三通道计算灰度值。
__global__ void bgrToGrayKernel(const unsigned char* bgr,  // __global__ 表示这是 CUDA kernel；bgr 指向 GPU 上的 BGR 输入图像数据。
                                unsigned char* gray,  // gray 指向 GPU 上的灰度输出缓冲区。
                                int width,  // width 是图像宽度，单位是像素。
                                int height,  // height 是图像高度，单位是像素。
                                int inputStepBytes) {  // inputStepBytes 是输入图像每一行实际占用的字节数。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程负责处理的像素横坐标。
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程负责处理的像素纵坐标。

    if (x >= width || y >= height) {  // 因为 grid 会向上取整，多出来的线程可能落在图像之外。
        return;  // 越界线程不访问显存，直接退出 kernel。
    }  // 结束越界判断。

    const int bgrIndex = y * inputStepBytes + x * 3;  // 根据行步长和三通道布局计算当前像素 B 通道的起始下标。
    const unsigned char b = bgr[bgrIndex + 0];  // 读取 B 通道值；OpenCV 彩色图默认通道顺序是 BGR。
    const unsigned char g = bgr[bgrIndex + 1];  // 读取 G 通道值。
    const unsigned char r = bgr[bgrIndex + 2];  // 读取 R 通道值。

    const float value = 0.114f * b + 0.587f * g + 0.299f * r;  // 按常用亮度权重把 BGR 转换成灰度浮点值。
    gray[y * width + x] = static_cast<unsigned char>(__float2int_rn(value));  // 四舍五入成整数并写入灰度输出图。
}  // 结束 bgrToGrayKernel。

// 这是“一颗线程处理多个像素”的 BGR 转灰度版本。
// 它把整张图像看成 width * height 个连续像素，每个线程处理 pixelsPerThread 个连续像素。
// 例如 pixelsPerThread = 4 时，全局线程 0 处理像素 0~3，全局线程 1 处理像素 4~7。
__global__ void bgrToGrayMultiPixelKernel(const unsigned char* bgr,  // bgr 指向 GPU 上的 BGR 输入图像数据。
                                          unsigned char* gray,  // gray 指向 GPU 上的灰度输出缓冲区。
                                          int width,  // width 是图像宽度，单位是像素。
                                          int height,  // height 是图像高度，单位是像素。
                                          int inputStepBytes,  // inputStepBytes 是输入图像每一行实际占用的字节数。
                                          int pixelsPerThread) {  // pixelsPerThread 表示每个线程循环处理几个像素。
    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程在一维 grid 中的全局编号。
    const int totalPixels = width * height;  // 计算整张图一共有多少个像素。
    const int firstPixel = globalThreadId * pixelsPerThread;  // 当前线程负责的第一个线性像素编号。

    for (int i = 0; i < pixelsPerThread; ++i) {  // 在线程内部循环处理多个连续像素。
        const int pixelIndex = firstPixel + i;  // 当前循环处理的线性像素编号。
        if (pixelIndex >= totalPixels) {  // 最后一批线程可能超出图像总像素数。
            return;  // 超出后直接退出当前线程。
        }  // 结束越界判断。

        const int x = pixelIndex % width;  // 把线性编号还原成图像横坐标。
        const int y = pixelIndex / width;  // 把线性编号还原成图像纵坐标。
        const int bgrIndex = y * inputStepBytes + x * 3;  // 根据行步长和 BGR 三通道布局计算输入下标。

        const unsigned char b = bgr[bgrIndex + 0];  // 读取 B 通道值。
        const unsigned char g = bgr[bgrIndex + 1];  // 读取 G 通道值。
        const unsigned char r = bgr[bgrIndex + 2];  // 读取 R 通道值。

        const float value = 0.114f * b + 0.587f * g + 0.299f * r;  // 按常用亮度权重把 BGR 转换成灰度值。
        gray[pixelIndex] = static_cast<unsigned char>(__float2int_rn(value));  // 写入灰度图对应线性位置。
    }  // 结束单线程内部循环。
}  // 结束 bgrToGrayMultiPixelKernel。

__device__ unsigned char pixelAt(const unsigned char* image,  // __device__ 表示该函数只能在 GPU 代码中调用。
                                 int x,  // x 是要读取的像素横坐标。
                                 int y,  // y 是要读取的像素纵坐标。
                                 int width) {  // width 用于把二维坐标换算成一维数组下标。
    return image[y * width + x];  // 按行优先布局读取 image[y][x] 的像素值。
}  // 结束 pixelAt 设备函数。

// 这是自写 CUDA kernel 版本的 Sobel 边缘检测。
// 它没有调用 OpenCV CUDA 模块，而是直接在 GPU 上实现 3x3 Sobel 计算。
__global__ void sobelKernel(const unsigned char* gray,  // gray 指向 GPU 上的 8 位单通道灰度输入图。
                            unsigned char* edges,  // edges 指向 GPU 上的 8 位单通道边缘输出图。
                            int width,  // width 是图像宽度，单位是像素。
                            int height) {  // height 是图像高度，单位是像素。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 根据 block 和 thread 索引计算当前线程的横坐标。
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 根据 block 和 thread 索引计算当前线程的纵坐标。

    if (x >= width || y >= height) {  // 过滤掉超出图像范围的多余线程。
        return;  // 越界线程不做任何处理。
    }  // 结束越界判断。

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {  // Sobel 需要 3x3 邻域，图像边界没有完整邻域。
        edges[y * width + x] = 0;  // 自写版本把边界像素的边缘强度固定设为 0。
        return;  // 边界像素处理完后直接返回。
    }  // 结束边界判断。

    const int tl = pixelAt(gray, x - 1, y - 1, width);  // 读取左上像素，tl 表示 top-left。
    const int tc = pixelAt(gray, x,     y - 1, width);  // 读取上中像素，tc 表示 top-center。
    const int tr = pixelAt(gray, x + 1, y - 1, width);  // 读取右上像素，tr 表示 top-right。
    const int ml = pixelAt(gray, x - 1, y,     width);  // 读取左中像素，ml 表示 middle-left。
    const int mr = pixelAt(gray, x + 1, y,     width);  // 读取右中像素，mr 表示 middle-right。
    const int bl = pixelAt(gray, x - 1, y + 1, width);  // 读取左下像素，bl 表示 bottom-left。
    const int bc = pixelAt(gray, x,     y + 1, width);  // 读取下中像素，bc 表示 bottom-center。
    const int br = pixelAt(gray, x + 1, y + 1, width);  // 读取右下像素，br 表示 bottom-right。

    const int gx = -tl + tr - 2 * ml + 2 * mr - bl + br;  // 使用 Sobel X 卷积核计算水平方向梯度。
    const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;  // 使用 Sobel Y 卷积核计算垂直方向梯度。
    const int absGx = gx < 0 ? -gx : gx;  // 取 X 方向梯度绝对值。
    const int absGy = gy < 0 ? -gy : gy;  // 取 Y 方向梯度绝对值。
    const int magnitude = absGx + absGy > 255 ? 255 : absGx + absGy;  // 用 |gx| + |gy| 近似梯度幅值，并饱和到 255。

    edges[y * width + x] = static_cast<unsigned char>(magnitude);  // 把计算出的边缘强度写入输出图对应像素。
}  // 结束 sobelKernel。

// 这个版本用于 cv::cuda::GpuMat。GpuMat 每行可能有对齐填充，所以输入和输出都要传 step。
__global__ void bgrToGrayGpuMatKernel(const unsigned char* bgr,  // bgr 指向 GpuMat 管理的 GPU BGR 输入图。
                                      unsigned char* gray,  // gray 指向 GpuMat 管理的 GPU 灰度输出图。
                                      int width,  // width 是有效图像宽度，不包含行尾填充。
                                      int height,  // height 是有效图像高度。
                                      size_t inputStepBytes,  // inputStepBytes 是 gpuInput.step，也就是输入每行真实字节跨度。
                                      size_t grayStepBytes) {  // grayStepBytes 是 gpuGray.step，也就是输出每行真实字节跨度。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程负责的像素横坐标。
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程负责的像素纵坐标。

    if (x >= width || y >= height) {  // 多出来的线程不访问图像外的内存。
        return;  // 直接退出当前线程。
    }  // 结束越界判断。

    const unsigned char* bgrRow = bgr + y * inputStepBytes;  // 定位到输入图第 y 行的起始地址。
    unsigned char* grayRow = gray + y * grayStepBytes;  // 定位到灰度图第 y 行的起始地址。
    const int bgrIndex = x * 3;  // BGR 三通道图中，第 x 个像素占 3 个字节。

    const unsigned char b = bgrRow[bgrIndex + 0];  // 读取 B 通道。
    const unsigned char g = bgrRow[bgrIndex + 1];  // 读取 G 通道。
    const unsigned char r = bgrRow[bgrIndex + 2];  // 读取 R 通道。

    const float value = 0.114f * b + 0.587f * g + 0.299f * r;  // 用 BGR 加权计算灰度值。
    grayRow[x] = static_cast<unsigned char>(__float2int_rn(value));  // 写入当前行第 x 个灰度像素。
}  // 结束 bgrToGrayGpuMatKernel。

__device__ unsigned char pixelAtStep(const unsigned char* image,  // image 指向可能带行对齐的 GPU 图像。
                                     int x,  // x 是像素横坐标。
                                     int y,  // y 是像素纵坐标。
                                     size_t stepBytes) {  // stepBytes 是这张图每行真实占用的字节数。
    return image[y * stepBytes + x];  // 用 step 跨行，而不是假设每行刚好等于 width 字节。
}  // 结束 pixelAtStep 设备函数。

// 这个 Sobel 版本同样用于 cv::cuda::GpuMat，输入和输出都按各自的 step 访问。
__global__ void sobelGpuMatKernel(const unsigned char* gray,  // gray 指向 GpuMat 管理的 GPU 灰度图。
                                  unsigned char* edges,  // edges 指向 GpuMat 管理的 GPU 边缘图。
                                  int width,  // width 是有效图像宽度。
                                  int height,  // height 是有效图像高度。
                                  size_t grayStepBytes,  // grayStepBytes 是灰度图每行真实字节跨度。
                                  size_t edgeStepBytes) {  // edgeStepBytes 是边缘图每行真实字节跨度。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程负责的像素横坐标。
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程负责的像素纵坐标。

    if (x >= width || y >= height) {  // 过滤掉图像范围外的线程。
        return;  // 越界线程直接退出。
    }  // 结束越界判断。

    unsigned char* edgeRow = edges + y * edgeStepBytes;  // 定位到输出边缘图第 y 行。
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {  // 边界没有完整 3x3 邻域。
        edgeRow[x] = 0;  // 边界像素固定为 0。
        return;  // 边界像素处理完直接返回。
    }  // 结束边界判断。

    const int tl = pixelAtStep(gray, x - 1, y - 1, grayStepBytes);  // 读取左上像素。
    const int tc = pixelAtStep(gray, x,     y - 1, grayStepBytes);  // 读取上中像素。
    const int tr = pixelAtStep(gray, x + 1, y - 1, grayStepBytes);  // 读取右上像素。
    const int ml = pixelAtStep(gray, x - 1, y,     grayStepBytes);  // 读取左中像素。
    const int mr = pixelAtStep(gray, x + 1, y,     grayStepBytes);  // 读取右中像素。
    const int bl = pixelAtStep(gray, x - 1, y + 1, grayStepBytes);  // 读取左下像素。
    const int bc = pixelAtStep(gray, x,     y + 1, grayStepBytes);  // 读取下中像素。
    const int br = pixelAtStep(gray, x + 1, y + 1, grayStepBytes);  // 读取右下像素。

    const int gx = -tl + tr - 2 * ml + 2 * mr - bl + br;  // 计算 X 方向 Sobel 梯度。
    const int gy = -tl - 2 * tc - tr + bl + 2 * bc + br;  // 计算 Y 方向 Sobel 梯度。
    const int absGx = gx < 0 ? -gx : gx;  // 取 X 梯度绝对值。
    const int absGy = gy < 0 ? -gy : gy;  // 取 Y 梯度绝对值。
    const int magnitude = absGx + absGy > 255 ? 255 : absGx + absGy;  // 计算并限制到 8 位范围。

    edgeRow[x] = static_cast<unsigned char>(magnitude);  // 写入当前像素的边缘强度。
}  // 结束 sobelGpuMatKernel。

void runCustomCudaPipeline(const cv::Mat& input,  // 运行自写 CUDA kernel 管线，输入是 CPU 内存中的 OpenCV Mat。
                           const std::string& edgeOutputPath,  // edgeOutputPath 是边缘图输出路径。
                           const std::string& grayOutputPath) {  // grayOutputPath 是灰度图输出路径。
    const int width = input.cols;  // 从输入 Mat 读取图像宽度。
    const int height = input.rows;  // 从输入 Mat 读取图像高度。
    const size_t pixelCount = static_cast<size_t>(width) * height;  // 计算像素总数，用 size_t 减少大图溢出风险。
    const size_t inputBytes = input.step * input.rows;  // 计算输入 BGR 图像总字节数，step 包含行对齐可能产生的填充。

    DeviceBuffer<unsigned char> dInput(inputBytes);  // 在 GPU 上申请原始 BGR 输入缓冲区，按字节存储。
    DeviceBuffer<unsigned char> dGray(pixelCount);  // 在 GPU 上申请灰度图缓冲区，每个像素 1 字节。
    DeviceBuffer<unsigned char> dEdges(pixelCount);  // 在 GPU 上申请边缘图缓冲区，每个像素 1 字节。

    CUDA_CHECK(cudaMemcpy(dInput.get(),  // 调用 cudaMemcpy，把 CPU 输入图复制到 GPU 输入缓冲区。
                          input.data,  // 源地址是 cv::Mat 在 CPU 内存中的数据指针。
                          inputBytes,  // 拷贝大小是整张输入图占用的字节数。
                          cudaMemcpyHostToDevice));  // 拷贝方向是 Host 到 Device，也就是 CPU 到 GPU。

    const dim3 block(16, 16);  // 每个 CUDA block 使用 16x16 个线程，总共 256 个线程。
    const dim3 grid((width + block.x - 1) / block.x,  // 计算横向 block 数量，用向上取整覆盖所有列。
                    (height + block.y - 1) / block.y);  // 计算纵向 block 数量，用向上取整覆盖所有行。

    bgrToGrayKernel<<<grid, block>>>(dInput.get(),  // 启动自写 BGR 转灰度 kernel，传入 GPU 输入图指针。
                                     dGray.get(),  // 传入 GPU 灰度输出缓冲区指针。
                                     width,  // 传入图像宽度。
                                     height,  // 传入图像高度。
                                     static_cast<int>(input.step));  // 传入输入 Mat 的行步长，帮助 kernel 正确跨行访问。
    CUDA_CHECK(cudaGetLastError());  // 检查 kernel 启动阶段是否出错。
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待灰度 kernel 执行完成，并捕获运行期间的异步错误。

    cv::Mat gray(height, width, CV_8UC1);  // 在 CPU 内存中创建 8 位单通道灰度图，用于接收 GPU 结果。
    CUDA_CHECK(cudaMemcpy(gray.data,  // 目标地址是 CPU 灰度 Mat 的数据区。
                          dGray.get(),  // 源地址是 GPU 灰度缓冲区。
                          dGray.bytes(),  // 拷贝大小是灰度缓冲区总字节数。
                          cudaMemcpyDeviceToHost));  // 拷贝方向是 Device 到 Host，也就是 GPU 到 CPU。
    writeImageOrThrow(grayOutputPath, gray);  // 把灰度图写到磁盘，失败时抛异常。

    sobelKernel<<<grid, block>>>(dGray.get(),  // 启动自写 Sobel kernel，输入是 GPU 灰度图。
                                 dEdges.get(),  // 输出是 GPU 边缘图缓冲区。
                                 width,  // 传入图像宽度。
                                 height);  // 传入图像高度。
    CUDA_CHECK(cudaGetLastError());  // 检查 Sobel kernel 启动阶段是否出错。
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待 Sobel kernel 执行完成，并检查运行期间错误。

    cv::Mat edges(height, width, CV_8UC1);  // 在 CPU 内存中创建 8 位单通道边缘图，用于接收 GPU 输出。
    CUDA_CHECK(cudaMemcpy(edges.data,  // 目标地址是 CPU 边缘图 Mat 的数据区。
                          dEdges.get(),  // 源地址是 GPU 边缘图缓冲区。
                          dEdges.bytes(),  // 拷贝大小是边缘图缓冲区总字节数。
                          cudaMemcpyDeviceToHost));  // 拷贝方向是 GPU 到 CPU。
    writeImageOrThrow(edgeOutputPath, edges);  // 把边缘图写到磁盘，失败时抛异常。
}  // 结束 runCustomCudaPipeline。

void runOpenCvCustomCudaPipeline(const cv::Mat& input,  // 运行 OpenCV GpuMat 加自写 CUDA kernel 的混合管线。
                                 const std::string& edgeOutputPath,  // edgeOutputPath 是边缘图输出路径。
                                 const std::string& grayOutputPath) {  // grayOutputPath 是灰度图输出路径。
    const int deviceCount = cv::cuda::getCudaEnabledDeviceCount();  // 查询 OpenCV CUDA 能看到的 CUDA 设备数量。
    if (deviceCount <= 0) {  // 如果 OpenCV 没有找到可用 CUDA 设备。
        throw std::runtime_error("OpenCV CUDA reports no CUDA-enabled device.");  // 抛出异常说明无法运行混合管线。
    }  // 结束设备数量判断。

    cv::cuda::setDevice(0);  // 选择第 0 块 CUDA 设备，保证 GpuMat 和自写 kernel 使用同一块 GPU。

    const int width = input.cols;  // 从输入 Mat 读取图像宽度。
    const int height = input.rows;  // 从输入 Mat 读取图像高度。

    cv::cuda::GpuMat gpuInput;  // 用 OpenCV CUDA 容器管理 GPU 输入图。
    cv::cuda::GpuMat gpuGray(height, width, CV_8UC1);  // 用 OpenCV CUDA 容器管理 GPU 灰度图。
    cv::cuda::GpuMat gpuEdges(height, width, CV_8UC1);  // 用 OpenCV CUDA 容器管理 GPU 边缘图。

    gpuInput.upload(input);  // 使用 OpenCV 的 upload 把 CPU Mat 上传到 GPU GpuMat。

    const dim3 block(16, 16);  // 每个 block 使用 16x16 个线程。
    const dim3 grid((width + block.x - 1) / block.x,  // 横向 block 数量向上取整。
                    (height + block.y - 1) / block.y);  // 纵向 block 数量向上取整。

    bgrToGrayGpuMatKernel<<<grid, block>>>(gpuInput.ptr<unsigned char>(),  // 传入 GpuMat 输入图的 GPU 指针。
                                           gpuGray.ptr<unsigned char>(),  // 传入 GpuMat 灰度图的 GPU 指针。
                                           width,  // 传入图像宽度。
                                           height,  // 传入图像高度。
                                           gpuInput.step,  // 使用 GPU 输入图自己的行步长。
                                           gpuGray.step);  // 使用 GPU 灰度图自己的行步长。
    CUDA_CHECK(cudaGetLastError());  // 检查灰度 kernel 是否成功启动。
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待灰度 kernel 执行完成并捕获异步错误。

    cv::Mat gray;  // 声明 CPU 灰度图 Mat，用于接收下载结果。
    gpuGray.download(gray);  // 使用 OpenCV 的 download 把 GPU 灰度图下载回 CPU。
    writeImageOrThrow(grayOutputPath, gray);  // 把灰度图写入磁盘。

    sobelGpuMatKernel<<<grid, block>>>(gpuGray.ptr<unsigned char>(),  // 传入 GpuMat 灰度图的 GPU 指针。
                                       gpuEdges.ptr<unsigned char>(),  // 传入 GpuMat 边缘图的 GPU 指针。
                                       width,  // 传入图像宽度。
                                       height,  // 传入图像高度。
                                       gpuGray.step,  // 使用 GPU 灰度图自己的行步长。
                                       gpuEdges.step);  // 使用 GPU 边缘图自己的行步长。
    CUDA_CHECK(cudaGetLastError());  // 检查 Sobel kernel 是否成功启动。
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待 Sobel kernel 执行完成并捕获异步错误。

    cv::Mat edges;  // 声明 CPU 边缘图 Mat，用于接收下载结果。
    gpuEdges.download(edges);  // 使用 OpenCV 的 download 把 GPU 边缘图下载回 CPU。
    writeImageOrThrow(edgeOutputPath, edges);  // 把边缘图写入磁盘。
}  // 结束 runOpenCvCustomCudaPipeline。

void runOpenCvCudaPipeline(const cv::Mat& input,  // 运行 OpenCV CUDA 管线，输入仍然是 CPU 内存中的 cv::Mat。
                           const std::string& edgeOutputPath,  // edgeOutputPath 是 OpenCV CUDA 边缘图输出路径。
                           const std::string& grayOutputPath) {  // grayOutputPath 是 OpenCV CUDA 灰度图输出路径。
    const int deviceCount = cv::cuda::getCudaEnabledDeviceCount();  // 查询 OpenCV CUDA 能看到的 CUDA 设备数量。
    if (deviceCount <= 0) {  // 如果 OpenCV 没有找到可用 CUDA 设备。
        throw std::runtime_error("OpenCV CUDA reports no CUDA-enabled device.");  // 抛出异常说明无法运行 OpenCV CUDA 路径。
    }  // 结束设备数量判断。

    cv::cuda::setDevice(0);  // 选择第 0 块 CUDA 设备作为 OpenCV CUDA 的执行设备。

    cv::cuda::GpuMat gpuInput;  // 声明 GPU 上的输入图像容器。
    cv::cuda::GpuMat gpuGray;  // 声明 GPU 上的灰度图容器。
    cv::cuda::GpuMat gpuGradX;  // 声明 GPU 上的 X 方向 Sobel 梯度图容器，类型为 16 位有符号。
    cv::cuda::GpuMat gpuGradY;  // 声明 GPU 上的 Y 方向 Sobel 梯度图容器，类型为 16 位有符号。
    cv::cuda::GpuMat gpuAbsX;  // 声明 GPU 上的 X 梯度绝对值图容器。
    cv::cuda::GpuMat gpuAbsY;  // 声明 GPU 上的 Y 梯度绝对值图容器。
    cv::cuda::GpuMat gpuMagnitude;  // 声明 GPU 上的梯度强度临时图容器。
    cv::cuda::GpuMat gpuEdges;  // 声明 GPU 上最终 8 位边缘图容器。

    gpuInput.upload(input);  // 把 CPU 内存中的 cv::Mat 上传到 GPU 的 cv::cuda::GpuMat。
    cv::cuda::cvtColor(gpuInput, gpuGray, cv::COLOR_BGR2GRAY);  // 调用 OpenCV CUDA 把 BGR 图转换成灰度图。

    cv::Ptr<cv::cuda::Filter> sobelX =  // 创建 X 方向 Sobel CUDA 滤波器智能指针。
        cv::cuda::createSobelFilter(CV_8UC1, CV_16SC1, 1, 0, 3);  // 输入 8 位单通道，输出 16 位有符号，dx=1、dy=0、核大小 3。
    cv::Ptr<cv::cuda::Filter> sobelY =  // 创建 Y 方向 Sobel CUDA 滤波器智能指针。
        cv::cuda::createSobelFilter(CV_8UC1, CV_16SC1, 0, 1, 3);  // 输入 8 位单通道，输出 16 位有符号，dx=0、dy=1、核大小 3。

    sobelX->apply(gpuGray, gpuGradX);  // 在 GPU 上应用 X 方向 Sobel 滤波器。
    sobelY->apply(gpuGray, gpuGradY);  // 在 GPU 上应用 Y 方向 Sobel 滤波器。
    cv::cuda::abs(gpuGradX, gpuAbsX);  // 在 GPU 上计算 X 梯度绝对值。
    cv::cuda::abs(gpuGradY, gpuAbsY);  // 在 GPU 上计算 Y 梯度绝对值。
    cv::cuda::add(gpuAbsX, gpuAbsY, gpuMagnitude, cv::noArray(), CV_16SC1);  // 在 GPU 上计算 |gx| + |gy|，保持 16 位结果避免中间溢出。
    gpuMagnitude.convertTo(gpuEdges, CV_8UC1);  // 把 16 位梯度强度饱和转换成 8 位边缘图。

    cv::Mat gray;  // 声明 CPU 内存中的灰度图 Mat，用于保存下载结果。
    cv::Mat edges;  // 声明 CPU 内存中的边缘图 Mat，用于保存下载结果。
    gpuGray.download(gray);  // 把 GPU 灰度图下载到 CPU 内存。
    gpuEdges.download(edges);  // 把 GPU 边缘图下载到 CPU 内存。

    zeroImageBorder(edges);  // 把边缘图边界置零，使输出边界策略更接近自写 Sobel kernel。
    writeImageOrThrow(grayOutputPath, gray);  // 把 OpenCV CUDA 生成的灰度图写到磁盘。
    writeImageOrThrow(edgeOutputPath, edges);  // 把 OpenCV CUDA 生成的边缘图写到磁盘。
}  // 结束 runOpenCvCudaPipeline。

void printUsage(const char* programName) {  // 打印命令行使用说明。
    std::cerr << "Usage:\n"  // 输出 usage 标题到标准错误。
              << "  " << programName  // 输出不带参数的形式，直接使用默认 input.png。
              << "\n"
              << "  " << programName  // 输出可执行文件名，方便用户直接复制当前命令格式。
              << " [input_image] [edge_output_image] [gray_output_image] [mode]\n"  // 输出完整参数形式。
              << "  " << programName << " [input_image] [mode]\n"
              << "  " << programName << " [mode]\n\n"  // 输出只指定模式、输入仍使用默认 input.png 的形式。
              << "defaults:\n"
              << "  input image:  ./input.png\n"
              << "  edges image:  cuda_edges.png\n"
              << "  gray image:   cuda_gray.png\n\n"
              << "mode:\n"  // 输出模式说明标题。
              << "  custom         Use self-written CUDA kernels.\n"  // 说明 custom 模式使用自写 CUDA kernel。
              << "  opencv-cuda    Use OpenCV CUDA functions.\n"  // 说明 opencv-cuda 模式使用 OpenCV CUDA 模块。
              << "  opencv-custom  Use OpenCV GpuMat upload/download with self-written CUDA kernels. This is the default.\n"  // 说明 opencv-custom 模式使用 GpuMat 加自写 kernel。
              << "  both         Run both implementations. OpenCV CUDA outputs use _opencv_cuda suffix.\n";  // 说明 both 模式会输出两套结果。
}  // 结束 printUsage 函数。

int main(int argc, char** argv) {  // 程序入口函数，argc 是参数数量，argv 是参数字符串数组。
    try {  // 用 try 包住主流程，统一捕获 OpenCV 异常和标准异常。
        if (argc > 5) {  // 参数数量最多支持 5 个，包含程序名本身。
            printUsage(argv[0]);  // 参数数量不合法时打印使用说明。
            return 1;  // 返回非 0，表示程序没有成功完成。
        }  // 结束参数数量判断。

        std::string inputPath = "./1.png";  // 默认从当前工作目录读取 input.png。
        if (argc >= 2) {
            inputPath = argv[1];
        }
        std::string edgeOutputPath = "cuda_edges.png";  // 默认边缘图输出文件名。
        std::string grayOutputPath = "cuda_gray.png";  // 默认灰度图输出文件名。
        std::string modeText = "opencv-custom";  // 默认运行 OpenCV GpuMat 加自写 CUDA kernel 的混合模式。

        if (argc >= 2) {  // 如果用户提供了至少一个参数，就解析命令行覆盖默认值。
            if (isModeText(argv[1])) {  // 如果第一个参数是模式名，就继续使用默认 input.png。
                if (argc > 2) {  // 只指定模式时不再接受额外参数，避免参数含义变得含糊。
                    printUsage(argv[0]);  // 打印用法提示用户改用完整参数格式。
                    return 1;  // 返回非 0，表示参数格式不合法。
                }  // 结束额外参数检查。
                modeText = argv[1];  // 把第一个参数作为运行模式。
            } else {  // 否则第一个参数仍然按输入图片路径解析。
                inputPath = argv[1];  // 用用户传入的图片路径覆盖默认 input.png。
                if (argc >= 3 && isModeText(argv[2])) {  // 如果第二个用户参数直接是模式名。
                    modeText = argv[2];  // 把第二个用户参数作为运行模式。
                } else {  // 否则按照传统格式解析输出路径和可选模式。
                    if (argc >= 3) {  // 如果提供了第二个用户参数。
                        edgeOutputPath = argv[2];  // 把第二个用户参数作为边缘图输出路径。
                    }  // 结束边缘图路径解析。
                    if (argc >= 4) {  // 如果提供了第三个用户参数。
                        grayOutputPath = argv[3];  // 把第三个用户参数作为灰度图输出路径。
                    }  // 结束灰度图路径解析。
                    if (argc == 5) {  // 如果提供了第四个用户参数。
                        modeText = argv[4];  // 把第四个用户参数作为运行模式。
                    }  // 结束模式参数解析。
                }  // 结束输出路径和模式解析。
            }  // 结束第一个参数类型判断。
        }  // 结束命令行参数布局判断。

        const PipelineMode mode = parseMode(modeText);  // 把模式字符串解析成枚举，非法模式会抛异常。

        cv::Mat input = cv::imread(inputPath, cv::IMREAD_COLOR);  // 用 OpenCV 从磁盘读取输入图像，强制读成 BGR 彩色图。
        if (input.empty()) {  // 如果读取失败，OpenCV 会返回空 Mat。
            throw std::runtime_error("Failed to read image: " + inputPath);  // 抛出异常并报告失败路径。
        }  // 结束读图失败判断。
        if (!input.isContinuous()) {  // 检查 Mat 数据是否连续，方便一次性上传或 cudaMemcpy。
            input = input.clone();  // 如果不连续，就复制出一份连续存储的 Mat。
        }  // 结束连续性检查。

        if (mode == PipelineMode::CustomCuda || mode == PipelineMode::Both) {  // 如果当前需要运行自写 CUDA kernel 路径。
            runCustomCudaPipeline(input, edgeOutputPath, grayOutputPath);  // 执行自写 CUDA kernel 图像处理管线。
        }  // 结束自写 CUDA 路径判断。

        if (mode == PipelineMode::OpenCvCustomCuda) {  // 如果当前需要运行 OpenCV GpuMat 加自写 CUDA kernel 路径。
            runOpenCvCustomCudaPipeline(input, edgeOutputPath, grayOutputPath);  // 执行 upload/download 加自写 kernel 的混合管线。
        }  // 结束混合 CUDA 路径判断。

        if (mode == PipelineMode::OpenCvCuda || mode == PipelineMode::Both) {  // 如果当前需要运行 OpenCV CUDA 路径。
            const std::string opencvEdgePath =  // 计算 OpenCV CUDA 边缘图输出路径。
                mode == PipelineMode::Both  // 如果是 both 模式，需要避免覆盖自写 CUDA 的输出文件。
                    ? addSuffixBeforeExtension(edgeOutputPath, "_opencv_cuda")  // both 模式下给边缘图文件名加 _opencv_cuda 后缀。
                    : edgeOutputPath;  // 非 both 模式下直接使用用户指定的边缘图路径。
            const std::string opencvGrayPath =  // 计算 OpenCV CUDA 灰度图输出路径。
                mode == PipelineMode::Both  // 如果是 both 模式，需要避免覆盖自写 CUDA 的输出文件。
                    ? addSuffixBeforeExtension(grayOutputPath, "_opencv_cuda")  // both 模式下给灰度图文件名加 _opencv_cuda 后缀。
                    : grayOutputPath;  // 非 both 模式下直接使用用户指定的灰度图路径。
            runOpenCvCudaPipeline(input, opencvEdgePath, opencvGrayPath);  // 执行 OpenCV CUDA 图像处理管线。
        }  // 结束 OpenCV CUDA 路径判断。

        std::cout << "Input:  " << inputPath << '\n'  // 输出输入图像路径。
                  << "Mode:   " << modeName(mode) << '\n'  // 输出本次运行模式。
                  << "Size:   " << input.cols << " x " << input.rows << '\n';  // 输出图像宽高。

        if (mode == PipelineMode::CustomCuda) {  // 如果只运行自写 CUDA kernel 模式。
            std::cout << "Gray:   " << grayOutputPath << '\n'  // 输出灰度图路径。
                      << "Edges:  " << edgeOutputPath << '\n';  // 输出边缘图路径。
        } else if (mode == PipelineMode::OpenCvCuda) {  // 如果只运行 OpenCV CUDA 模式。
            std::cout << "Gray:   " << grayOutputPath << '\n'  // 输出灰度图路径。
                      << "Edges:  " << edgeOutputPath << '\n';  // 输出边缘图路径。
        } else if (mode == PipelineMode::OpenCvCustomCuda) {  // 如果只运行 OpenCV GpuMat 加自写 CUDA kernel 模式。
            std::cout << "Gray:   " << grayOutputPath << '\n'  // 输出灰度图路径。
                      << "Edges:  " << edgeOutputPath << '\n';  // 输出边缘图路径。
        } else {  // 否则就是 both 模式，需要分别输出两套文件路径。
            std::cout << "Custom gray:       " << grayOutputPath << '\n'  // 输出自写 CUDA 灰度图路径。
                      << "Custom edges:      " << edgeOutputPath << '\n'  // 输出自写 CUDA 边缘图路径。
                      << "OpenCV CUDA gray:  "  // 输出 OpenCV CUDA 灰度图标签。
                      << addSuffixBeforeExtension(grayOutputPath, "_opencv_cuda") << '\n'  // 输出 OpenCV CUDA 灰度图路径。
                      << "OpenCV CUDA edges: "  // 输出 OpenCV CUDA 边缘图标签。
                      << addSuffixBeforeExtension(edgeOutputPath, "_opencv_cuda") << '\n';  // 输出 OpenCV CUDA 边缘图路径。
        }  // 结束输出路径分支。

        return 0;  // 主流程成功完成，返回 0。
    } catch (const cv::Exception& ex) {  // 捕获 OpenCV 抛出的异常，例如 CUDA 模块运行失败或图像编码失败。
        std::cerr << "OpenCV error: " << ex.what() << '\n';  // 输出 OpenCV 异常详情。
        return 1;  // 返回非 0 表示程序失败。
    } catch (const std::exception& ex) {  // 捕获标准 C++ 异常，例如 runtime_error。
        std::cerr << ex.what() << '\n';  // 输出异常文本。
        return 1;  // 返回非 0 表示程序失败。
    }  // 结束异常处理。
}  // 结束 main 函数。
