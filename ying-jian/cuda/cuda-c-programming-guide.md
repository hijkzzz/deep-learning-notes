# CUDA C Programming Guide

## Introduction

### From Graphics Processing to General Purpose Parallel Computing

在可实现的高清3D图形市场需求的推动下，可编程图形处理器单元或GPU已经发展成为高度并行，多线程，多核处理器，具有巨大的计算能力和非常高的内存带宽，如图1和图所示 2。

Figure 1. Floating-Point Operations per Second for the CPU and GPU

![](../../.gitbook/assets/image%20%28223%29.png)

Figure 2. Memory Bandwidth for the CPU and GPU

![](../../.gitbook/assets/image%20%28140%29.png)

CPU和GPU之间浮点能力差异背后的原因是GPU专门用于计算密集型，高度并行计算 - 正是图形渲染的关键 - 因此设计使得更多晶体管用于数据处理 而不是数据缓存和流量控制，如图3示意性所示。

Figure 3. The GPU Devotes More Transistors to Data Processing

![](../../.gitbook/assets/image%20%2840%29.png)

更具体地说，图形处理器特别适合于解决可以表示为数据并行计算的问题——在许多数据元素上并行执行相同的程序——具有高运算强度——算术运算与存储器运算的比率。因为对每个数据元素执行相同的程序，所以对复杂的流控制的要求较低，并且因为它在许多数据元素上执行并且具有高运算强度，所以可以用计算而不是大数据高速缓存来隐藏存储器访问延迟。

数据并行处理将数据元素映射到并行处理线程。许多处理大型数据集的应用程序可以使用数据并行编程模型来加速计算。在3D渲染中，大量像素和顶点被映射到并行线程。类似地，图像和媒体处理应用，例如渲染图像的后处理、视频编码和解码、图像缩放、立体视觉和模式识别，可以将图像块和像素映射到并行处理线程。事实上，从一般信号处理或物理模拟到计算金融或计算生物学，图像渲染和处理领域之外的许多算法都通过数据并行处理得到了加速。

### CUDA®: A General-Purpose Parallel Computing Platform and Programming Model

 2006年11月，NVIDIA推出了CUDA®，这是一种通用并行计算平台和编程模型，它利用NVIDIA GPU中的并行计算引擎，以比CPU更高效的方式解决许多复杂的计算问题。

CUDA带有一个软件环境，允许开发人员使用C作为高级编程语言。 如图4所示，支持其他语言，应用程序编程接口或基于指令的方法，例如FORTRAN，DirectCompute，OpenACC。

Figure 4. GPU Computing Applications. CUDA is designed to support various languages and application programming interfaces.

![](../../.gitbook/assets/image%20%28161%29.png)

### A Scalable Programming Model

多核CPU和多核GPU的出现意味着主流处理器芯片现在是并行系统。 面临的挑战是开发透明地扩展其并行性的应用软件，以利用越来越多的处理器内核，就像3D图形应用程序透明地将其并行性扩展到具有大量不同内核的多核GPU一样。

CUDA并行编程模型旨在克服这一挑战，同时为熟悉标准编程语言（如C）的程序员保持较低的学习曲线。

其核心是三个关键的抽象 - 线程组，共享存储器和屏障同步的层次结构 - 它们只是作为最小的语言扩展集向程序员公开。

这些抽象提供了细粒度数据并行性和线程并行性，嵌套在粗粒度数据并行和任务并行中。 它们指导程序员将问题划分为粗略的子问题，这些子问题可以通过线程块并行地独立解决，并且每个子问题都可以更精细，可以由块内的所有线程并行地协同解决。

这种分解通过允许线程在解决每个子问题时进行协作来保持语言表达能力，同时实现自动可伸缩性。 实际上，每个线程块可以在GPU内的任何可用多处理器上以任何顺序，同时或顺序调度，以便编译的CUDA程序可以在任何数量的多处理器上执行，如图5所示，并且仅运行时 系统需要知道物理多处理器计数。

这种可扩展的编程模型允许GPU架构通过简单地扩展多处理器和内存分区的数量来跨越广泛的市场范围：从高性能发烧友GeForce GPU和专业Quadro和Tesla计算产品到各种廉价的主流GeForce GPU（ 请参阅支持CUDA的GPU以获取所有支持CUDA的GPU的列表。

Figure 5. Automatic Scalability

![](../../.gitbook/assets/image%20%28218%29.png)

### Document Structure

本文档分为以下章节：

* 章节 [Introduction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction) is a general introduction to CUDA.
* 章节 [Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) outlines the CUDA programming model.
* 章节 [Programming Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface) describes the programming interface.
* 章节 [Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation) describes the hardware implementation.
* 章节 [Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines) gives some guidance on how to achieve maximum performance.
* 附录 [CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus) lists all CUDA-enabled devices.
* 附录 [C Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions) is a detailed description of all extensions to the C language.
* 附录 [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) describes synchronization primitives for various groups of CUDA threads.
* 附录 [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism) describes how to launch and synchronize one kernel from another.
* 附录 [Mathematical Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix) lists the mathematical functions supported in CUDA.
* 附录 [C/C++ Language Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support) lists the C++ features supported in device code.
* 附录 [Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching) gives more details on texture fetching
* 附录 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) gives the technical specifications of various devices, as well as more architectural details.
* 附录 [Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api) introduces the low-level driver API.
* 附录 [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) lists all the CUDA environment variables.
* 附录 [Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd) introduces the Unified Memory programming model.

## Programming Model

本章通过概述CUDA编程模型在C语言中的公开方式，介绍了CUDA编程模型背后的主要概念。

本章和下一章中使用的向量加法示例的完整代码可以在`vectorAdd` CUDA示例中找到。

### Kernels

CUDA C通过允许程序员定义称为内核的C函数来扩展C，这些函数在被调用时由N个不同的CUDA线程并行执行N次，而不是像常规C函数那样只执行一次。

使用`__global__`声明说明符定义内核，并使用新的`<<< ... >>>`执行配置语法指定为给定内核调用执行该内核的CUDA线程数（请参阅C语言扩展）。 执行内核的每个线程都有一个唯一的线程ID，可以通过内置的`threadIdx`变量在内核中访问。

```c
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

这里，执行VecAdd\(\)的N个线程中的每一个执行一对成对添加。

### Thread Hierarchy

为方便起见，threadIdx是一个3分量向量，因此可以使用一维，二维或三维线程索引来识别线程，从而形成一维，二维或三维块。 线程，称为线程块。 这提供了一种自然的方式来调用域中元素（如向量，矩阵或体积）的计算。

线程的索引及其线程ID以直接的方式相互关联：对于一维块，它们是相同的;对于二维块大小 $$\left(D_{x}, D_{y}\right)$$ ，索引线程的线程ID是 $$\left(x+y D_{x}\right)$$ ，对于三维块大小 $$\left(D_{X}, D_{V}, D_{Z}\right)$$ ，索引线程的线程ID是 $$\left(x+y D_{x}+z D_{x} D_{y}\right)$$ 。

作为示例，以下代码添加两个大小为N×N的矩阵A和B，并将结果存储到矩阵C中

```c
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

每个块的线程数有限制，因为块的所有线程都应该驻留在同一个处理器核心上，并且必须共享该核心的有限内存资源。 在当前的GPU上，线程块最多可包含1024个线程。

但是，内核可以由多个同形状的线程块执行，因此线程总数等于每个块的线程数乘以块数。

块被组织成一维，二维或三维线程块网格，如图6所示。网格中线程块的数量通常由正在处理的数据的大小或者数量决定。 系统中的处理器，它可以大大超过。

Figure 6. Grid of Thread Blocks

![](../../.gitbook/assets/image%20%2830%29.png)

每个块的线程数和&lt;&lt;&lt; ... &gt;&gt;&gt;语法中指定的每个网格的块数可以是int或dim3类型。 可以如上例中那样指定二维块或网格。

网格中的每个块可以通过内核中通过内置的blockIdx变量访问的一维，二维或三维索引来识别。 线程块的维度可以通过内置的blockDim变量在内核中访问。

扩展先前的MatAdd\(\)示例以处理多个块，代码如下所示。

```c
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

 线程块大小为16x16（256个线程），虽然在这种情况下是任意的，但却是常见的选择。 使用足够的块创建网格，以便像以前一样为每个矩阵元素创建一个线程。 为简单起见，此示例假定每个维度中每个网格的线程数可以被该维度中每个块的线程数整除，但不一定是这种情况。

线程块需要独立执行：必须能够以任何顺序，并行或串行执行它们。 这种独立性要求允许线程块以任意顺序在任意数量的内核上进行调度，如图5所示，使程序员能够编写随内核数量扩展的代码。

块内的线程可以通过一些共享内存共享数据并通过同步它们的执行来协调内存访问来协作。 更确切地说，可以通过调用 $$__syncthreads()$$ 内部函数来指定内核中的同步点; ****$$__syncthreads()$$充当一个屏障，在该屏障中，块中的所有线程必须等待才能允许任何线程继续。 共享内存提供了使用共享内存的示例。 除$$__syncthreads()$$之外，协作组API还提供了一组丰富的线程同步原语。

为了实现高效的协作，共享内存应该是每个处理器内核附近的低延迟内存\(非常像L1缓存\)，并且\_\_syncthreads\(\)应该是轻量级的

### Memory Hierarchy

CUDA线程可以在执行期间从多个内存空间访问数据，如图7所示。每个线程都有私有本地内存。 每个线程块都具有对块的所有线程可见的共享内存，并且具有与块相同的生存期。 所有线程都可以访问相同的全局内存。

所有线程都可以访问两个额外的只读内存空间：常量和纹理内存空间。 全局，常量和纹理内存空间针对不同的内存使用进行了优化（请参阅设备内存访问）。 纹理存储器还为某些特定数据格式提供不同的寻址模式以及数据滤波（请参阅[纹理和表面存储器](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)）。

全局，常量和纹理内存空间在同一应用程序的内核启动之间是持久的。

Figure 7. Memory Hierarchy

![](../../.gitbook/assets/image%20%28222%29.png)

### Heterogeneous Programming

如图8所示，CUDA编程模型假设CUDA线程在物理上独立的设备上执行，该设备作为运行C程序的主机的协处理器运行。 例如，当内核在GPU上执行而其余的C程序在CPU上执行时就是这种情况。

CUDA编程模型还假设主机和设备都在DRAM中保持它们自己独立的存储空间，分别称为主机存储器和设备存储器。 因此，程序通过调用CUDA运行时（[在编程接口中描述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)）来管理内核可见的全局，常量和纹理内存空间。 这包括设备内存分配和释放以及主机和设备内存之间的数据传输。

Unified Memory提供托管内存以桥接主机和设备内存空间。 可以从系统中的所有CPU和GPU访问托管内存，作为具有公共地址空间的单个连贯内存映像。 此功能可以实现设备内存的超额预订，并且无需在主机和设备上显式镜像数据，从而大大简化了移植应用程序的任务。 有关统一内存的介绍，请参阅[统一内存编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)。

Figure 8. Heterogeneous Programming

![](../../.gitbook/assets/image%20%2818%29.png)

注意：串行代码在主机上执行，而并行代码在设备上执行。

### Compute Capability

设备的计算能力由版本号表示，有时也称为“SM版本”。 此版本号标识GPU硬件支持的功能，并由运行时的应用程序用于确定当前GPU上可用的硬件功能和/或指令。

计算能力包括主修订号X和次修订号Y，并由X.Y表示。

具有相同主要修订号的设备具有相同的核心体系结构。 基于Volta架构的设备的主要版本号为7，基于Pascal架构的设备为6，基于Maxwell架构的设备为5，基于Kepler架构的设备为3，基于Fermi架构的设备为2， 和1为基于特斯拉架构的设备。

次要修订号对应于核心架构的增量改进，可能包括新功能。

图灵是计算能力7.5设备的架构，是基于Volta架构的增量更新。

[启用CUDA的GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)列出了所有支持CUDA的设备及其计算功能。 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)提供每种计算能力的技术规范。

注意：不应将特定GPU的计算能力版本与CUDA版本（例如，CUDA 7.5，CUDA 8，CUDA 9）混淆，后者是CUDA软件平台的版本。 应用程序开发人员使用CUDA平台创建在多代GPU架构上运行的应用程序，包括尚未发明的未来GPU架构。 虽然新版本的CUDA平台通常通过支持该架构的计算能力版本来添加对新GPU架构的本机支持，但新版本的CUDA平台通常还包括独立于硬件生成的软件功能。

从CUDA 7.0和CUDA 9.0开始，不再支持Tesla和Fermi架构。

## Programming Interface

CUDA C为熟悉C编程语言的用户提供了一条简单的路径，可以轻松编写程序以供设备执行。

它由对C语言的最小扩展集和运行时库组成。

核心语言扩展已在编程模型中介绍。它们允许程序员将内核定义为C函数，并在每次调用函数时使用一些新语法来指定网格和块维度。可以在[C语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)中找到所有扩展的完整描述。必须使用nvcc编译包含其中某些扩展的任何源文件，如使用[NVCC编译中所述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)。

运行时在[编译工作流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-workflow)中介绍。 它提供在主机上执行的C函数，用于分配和释放设备内存，在主机内存和设备内存之间传输数据，管理具有多个设备的系统等。可以在CUDA参考手册中找到运行时的完整描述。

运行时构建在较低级别的C API（CUDA驱动程序API）之上，该API也可由应用程序访问。驱动程序API通过暴露较低级别的概念（例如CUDA上下文 - 设备的主机进程的模拟）和CUDA模块（设备的动态加载库的模拟）来提供额外的控制级别。大多数应用程序不使用驱动程序API，因为它们不需要这种额外的控制级别，并且在使用运行时时，上下文和模块管理是隐式的，从而产生更简洁的代码。 驱动程序API在[Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)中介绍，并在参考手册中有详细描述。

### Compilation with NVCC

可以使用称为PTX的CUDA指令集架构来编写内核，这在PTX参考手册中有所描述。 然而，使用诸如C的高级编程语言通常更有效。在这两种情况下，必须通过nvcc将内核编译成二进制代码以在设备上执行。

nvcc是一个编译器驱动程序，它简化了编译C或PTX代码的过程：它提供了简单而熟悉的命令行选项，并通过调用实现不同编译阶段的工具集来执行它们。 本节概述了nvcc工作流和命令选项。 完整的描述可以在nvcc用户手册中找到。

## Hardware Implementation

## Performance Guidelines

## Appendix

* 附录 [CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus) lists all CUDA-enabled devices.
* 附录 [C Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions) is a detailed description of all extensions to the C language.
* 附录 [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) describes synchronization primitives for various groups of CUDA threads.
* 附录 [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism) describes how to launch and synchronize one kernel from another.
* 附录 [Mathematical Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix) lists the mathematical functions supported in CUDA.
* 附录 [C/C++ Language Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-cplusplus-language-support) lists the C++ features supported in device code.
* 附录 [Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching) gives more details on texture fetching
* 附录 [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities) gives the technical specifications of various devices, as well as more architectural details.
* 附录 [Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api) introduces the low-level driver API.
* 附录 [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars) lists all the CUDA environment variables.
* 附录 [Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd) introduces the Unified Memory programming model.

## 



