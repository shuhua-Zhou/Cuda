# 修改说明

## 变更内容

- 为 `src/cuda_image_pipeline.cu` 添加了详细中文注释，覆盖 CUDA 错误检查宏、GPU 显存管理、BGR 转灰度 kernel、Sobel kernel、CPU/GPU 数据拷贝和 OpenCV 读写图流程。
- 保持原有算法、函数签名和构建配置不变，只增强代码可读性和学习说明。
- 源码中未添加 `%TSD-Header-###%` 保护头。

## 编译验证

- 使用现有 `CudaOpenCvImagePipeline.sln` 和 Release x64 配置完成编译。
- 输出文件：`x64/Release/cuda_image_pipeline.exe`
- 编译结果：0 个警告，0 个错误。

## 注意事项

- 未跟踪文件 `cuda_gray.png` 是本地运行产物，本次提交未包含。
