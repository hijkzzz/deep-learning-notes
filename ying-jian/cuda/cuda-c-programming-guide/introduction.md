# Introduction



### From Graphics Processing to General Purpose Parallel Computing

在可实现的高清3D图形市场需求的推动下，可编程图形处理器单元或GPU已经发展成为高度并行，多线程，多核处理器，具有巨大的计算能力和非常高的内存带宽，如图1和图所示 2。

Figure 1. Floating-Point Operations per Second for the CPU and GPU

![](../../../.gitbook/assets/image%20%28238%29.png)

Figure 2. Memory Bandwidth for the CPU and GPU

![](../../../.gitbook/assets/image%20%28148%29.png)

CPU和GPU之间浮点能力差异背后的原因是GPU专门用于计算密集型，高度并行计算 - 正是图形渲染的关键 - 因此设计使得更多晶体管用于数据处理 而不是数据缓存和流量控制，如图3示意性所示。

Figure 3. The GPU Devotes More Transistors to Data Processing

![](../../../.gitbook/assets/image%20%2845%29.png)

更具体地说，图形处理器特别适合于解决可以表示为数据并行计算的问题——在许多数据元素上并行执行相同的程序——具有高运算强度——算术运算与存储器运算的比率。因为对每个数据元素执行相同的程序，所以对复杂的流控制的要求较低，并且因为它在许多数据元素上执行并且具有高运算强度，所以可以用计算而不是大数据高速缓存来隐藏存储器访问延迟。

数据并行处理将数据元素映射到并行处理线程。许多处理大型数据集的应用程序可以使用数据并行编程模型来加速计算。在3D渲染中，大量像素和顶点被映射到并行线程。类似地，图像和媒体处理应用，例如渲染图像的后处理、视频编码和解码、图像缩放、立体视觉和模式识别，可以将图像块和像素映射到并行处理线程。事实上，从一般信号处理或物理模拟到计算金融或计算生物学，图像渲染和处理领域之外的许多算法都通过数据并行处理得到了加速。

### CUDA®: A General-Purpose Parallel Computing Platform and Programming Model

 2006年11月，NVIDIA推出了CUDA®，这是一种通用并行计算平台和编程模型，它利用NVIDIA GPU中的并行计算引擎，以比CPU更高效的方式解决许多复杂的计算问题。

CUDA带有一个软件环境，允许开发人员使用C作为高级编程语言。 如图4所示，支持其他语言，应用程序编程接口或基于指令的方法，例如FORTRAN，DirectCompute，OpenACC。

Figure 4. GPU Computing Applications. CUDA is designed to support various languages and application programming interfaces.

![](../../../.gitbook/assets/image%20%28171%29.png)

### A Scalable Programming Model

多核CPU和多核GPU的出现意味着主流处理器芯片现在是并行系统。 面临的挑战是开发透明地扩展其并行性的应用软件，以利用越来越多的处理器内核，就像3D图形应用程序透明地将其并行性扩展到具有大量不同内核的多核GPU一样。

CUDA并行编程模型旨在克服这一挑战，同时为熟悉标准编程语言（如C）的程序员保持较低的学习曲线。

其核心是三个关键的抽象 - 线程组，共享存储器和屏障同步的层次结构 - 它们只是作为最小的语言扩展集向程序员公开。

这些抽象提供了细粒度数据并行性和线程并行性，嵌套在粗粒度数据并行和任务并行中。 它们指导程序员将问题划分为粗略的子问题，这些子问题可以通过线程块并行地独立解决，并且每个子问题都可以更精细，可以由块内的所有线程并行地协同解决。

这种分解通过允许线程在解决每个子问题时进行协作来保持语言表达能力，同时实现自动可伸缩性。 实际上，每个线程块可以在GPU内的任何可用多处理器上以任何顺序，同时或顺序调度，以便编译的CUDA程序可以在任何数量的多处理器上执行，如图5所示，并且仅运行时 系统需要知道物理多处理器计数。

这种可扩展的编程模型允许GPU架构通过简单地扩展多处理器和内存分区的数量来跨越广泛的市场范围：从高性能发烧友GeForce GPU和专业Quadro和Tesla计算产品到各种廉价的主流GeForce GPU（ 请参阅支持CUDA的GPU以获取所有支持CUDA的GPU的列表。

Figure 5. Automatic Scalability

![](../../../.gitbook/assets/image%20%28233%29.png)

### Document Structure

本文档分为以下章节：

* 章节 [Introduction](introduction.md) 是对CUDA的一般介绍。
* 章节 [Programming Model](programming-model.md) 概述了CUDA编程模型。
* 章节 [Programming Interface](programming-interface.md) 描述了编程接口。
* 章节 [Hardware Implementation](hardware-implementation.md) 描述了硬件实现。
* 章节 [Performance Guidelines](performance-guidelines.md) 为如何实现最佳性能提供了一些指导。
* 附录 [CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus) 列出了所有支持CUDA的设备。
* 附录 [C Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions) 是对C语言的所有扩展的详细描述。
* 附录 [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) 描述了各种CUDA线程组的同步原语。
* 附录 [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism) 描述了如何从另一个内核启动和同步一个内核。
* 附录 [Mathematical Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix)列出了CUDA中支持的数学函数。
* 附录 [C/C++ Language Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support)列出了设备代码中支持的C ++功能。
* 附录 [Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching) 提供了有关纹理提取的更多细节
* 附录 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) 给出了各种设备的技术规范，以及更多的架构细节。
* 附录 [Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api) 介绍了低级驱动程序API。
* 附录 [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) 列出了所有CUDA环境变量。
* 附录 [Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd) 介绍了统一内存编程模型。

