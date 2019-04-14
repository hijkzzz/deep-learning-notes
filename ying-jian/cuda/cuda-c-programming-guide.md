# CUDA C Programming Guide

## Introduction

### From Graphics Processing to General Purpose Parallel Computing

在可实现的高清3D图形市场需求的推动下，可编程图形处理器单元或GPU已经发展成为高度并行，多线程，多核处理器，具有巨大的计算能力和非常高的内存带宽，如图1和图所示 2。

Figure 1. Floating-Point Operations per Second for the CPU and GPU

![](../../.gitbook/assets/image%20%28225%29.png)

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

![](../../.gitbook/assets/image%20%28220%29.png)

### Document Structure

本文档分为以下章节：

* 章节 [Introduction](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction) 是对CUDA的一般介绍。
* 章节 [Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) 概述了CUDA编程模型。
* 章节 [Programming Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface) 描述了编程接口。
* 章节 [Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation) 描述了硬件实现。
* 章节 [Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines) 为如何实现最佳性能提供了一些指导。
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

![](../../.gitbook/assets/image%20%28224%29.png)

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

#### Compilation Workflow

**Offline Compilation**

用nvcc编译的源文件可以包括主机代码（即，在主机上执行的代码）和设备代码（即，在设备上执行的代码）的混合。 nvcc的基本工作流程包括将设备代码与主机代码分离，然后：

* 将设备代码编译为汇编表（PTX代码）和/或二进制表（cubin对象），
* 通过必要的CUDA C运行时函数调用替换内核中引入的&lt;&lt;&lt; ... &gt;&gt;&gt;语法（并在执行配置中更详细地描述）来修改主机代码，以从PTX代码加载和启动每个编译的内核 和/或cubin对象。

修改后的主机代码既可以作为C代码输出，也可以使用其他工具进行编译，也可以通过让nvcc在上一个编译阶段调用主机编译器直接输出目标代码。

应用程序可以：

* 链接到已编译的主机代码（这是最常见的情况），
* 或者忽略修改后的主机代码（如果有）并使用CUDA驱动程序API（请参阅驱动程序API）来加载和执行PTX代码或cubin对象

**Just-in-Time Compilation**

应用程序在运行时加载的任何PTX代码都由设备驱动程序进一步编译为二进制代码。 这称为即时编译。 即时编译会增加应用程序加载时间，但允许应用程序受益于每个新设备驱动程序随附的任何新编译器改进。 它也是应用程序在编译应用程序时不存在的设备上运行的唯一方法，如应用程序兼容性中所述。

当设备驱动程序即时编译某些应用程序的某些PTX代码时，它会自动缓存生成的二进制代码的副本，以避免在后续应用程序调用中重复编译。 缓存（称为计算缓存）在升级设备驱动程序时自动失效，因此应用程序可以从设备驱动程序中内置的新实时编译器的改进中受益。

环境变量可用于控制即时编译，如CUDA环境变量中所述

#### Binary Compatibility

二进制代码是特定于体系结构的。 使用编译器选项-code生成cubin对象，该选项指定目标体系结构：例如，使用-code = sm\_35进行编译会为计算能力3.5的设备生成二进制代码。 从一个小修订版到下一个修订版保证二进制兼容性，但不是从一个小修订版到前一个修订版或主要修订版。 换句话说，为计算能力X.y生成的cubin对象将仅在计算能力X.z的设备上执行，其中z≥y。

注意：仅桌面支持二进制兼容性。 Tegra不支持它。 此外，不支持桌面和Tegra之间的二进制兼容性。

#### PTX Compatibility

某些PTX指令仅在具有更高计算能力的设备上受支持。 例如，Warp Shuffle Functions仅在计算能力3.0及以上的设备上受支持。 -arch编译器选项指定在将C编译为PTX代码时假定的计算能力。 因此，包含warp shuffle的代码必须使用-arch = compute\_30（或更高版本）进行编译。

为某些特定计算能力生成的PTX代码始终可以编译为具有更大或相等计算能力的二进制代码。 请注意，从早期PTX版本编译的二进制文件可能无法使用某些硬件功能。 例如，从为计算能力6.0（Pascal）生成的PTX编译的计算能力7.0（Volta）的二进制目标设备将不使用Tensor Core指令，因为这些指令在Pascal上不可用。 结果，如果使用最新版本的PTX生成二进制文件，则最终二进制文件可能表现得更差。

#### Application Compatibility

要在具有特定计算能力的设备上执行代码，应用程序必须加载与此计算功能兼容的二进制或PTX代码，如二进制兼容性和PTX兼容性中所述。 特别是，为了能够在具有更高计算能力的未来架构上执行代码（尚未生成二进制代码），应用程序必须加载将为这些设备及时编译的PTX代码（请参阅Just In Time编译）。

嵌入在CUDA C应用程序中的PTX和二进制代码由-arch和-code编译器选项或-gencode编译器选项控制，如nvcc用户手册中所述。 例如，

```bash
nvcc x.cu
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=\'compute_60,sm_60\'
```

嵌入与计算能力3.5和5.0兼容的二进制代码（第一和第二代码选项）和兼容计算能力6.0的PTX和二进制代码（第三代码选项）

生成主机代码以在运行时自动选择要加载和执行的最合适的代码，在上面的示例中，将是：

* 3.5 binary code for devices with compute capability 3.5 and 3.7,
* 5.0 binary code for devices with compute capability 5.0 and 5.2,
* 6.0 binary code for devices with compute capability 6.0 and 6.1,
* PTX code which is compiled to binary code at runtime for devices with compute capability 7.0 and higher.

例如，x.cu可以具有使用warp shuffle操作的优化代码路径，仅在计算能力3.0及更高版本的设备中支持。\_\_CUDA\_ARCH\_\_宏可用于根据计算能力区分各种代码路径。 它仅针对设备代码定义。 例如，当使用-arch = compute_35进行编译时，\_\_ CUDA\_ARCH\_\__等于350。

#### C/C++ Compatibility

编译器的前端根据C ++语法规则处理CUDA源文件。 主机代码支持完整的C ++。 但是，如C / C ++语言支持中所述，设备代码仅完全支持C ++的一个子集。

#### 64-Bit Compatibility

64位版本的nvcc以64位模式编译设备代码（即指针是64位）。 只有在64位模式下编译的主机代码才支持以64位模式编译的器件代码。

类似地，32位版本的nvcc以32位模式编译器件代码，而以32位模式编译的器件代码仅支持以32位模式编译的主机代码。

32位版本的nvcc也可以使用-m64编译器选项以64位模式编译设备代码。

64位版本的nvcc也可以使用-m32编译器选项以32位模式编译设备代码。

### CUDA C Runtime

运行时在cudart库中实现，该库通过cudart.lib或libcudart.a静态链接到应用程序，或通过cudart.dll或libcudart.so动态链接。 需要cudart.dll和/或cudart.so进行动态链接的应用程序通常将它们作为应用程序安装包的一部分包含在内。 在链接到CUDA运行时的同一实例的组件之间传递CUDA运行时符号的地址是安全的。

所有入口点都以cuda为前缀。

正如 [Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming) 中提到的, CUDA编程模型假设一个系统由一个主机和一个设备组成，每个设备都有各自独立的内存。 [Device Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)概述了用于管理设备内存的运行时函数。

[Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) 说明了在线程层次结构中引入的共享内存的使用，以最大限度地提高性能。

[Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory) 引入页面锁定的主机内存，它需要将内核执行与主机和设备内存之间的数据传输重叠。

[Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution) 描述了用于在系统中的各个级别启用异步并发执行的概念和API。

[Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system) 显示了编程模型如何扩展到具有连接到同一主机的多个设备的系统。

[Error Checking](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking) 描述了如何正确检查运行时生成的错误。

[Call Stack](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#call-stack) 提到用于管理CUDA C调用堆栈的运行时函数。

[Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)呈现纹理和表面存储空间，提供访问设备存储器的另一种方式; 它们还暴露了GPU纹理硬件的一个子集。

[Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphics-interoperability) 介绍了运行时提供的各种功能，以便与两个主要的图形API，OpenGL和Direct3D进行互操作。

#### Initialization

运行时没有明确的初始化函数; 它在第一次调用运行时函数时初始化（更具体地说，除了参考手册的设备和版本管理部分中的函数之外的任何函数）。 在计时运行时函数调用和从第一次调用运行时解释错误代码时，需要记住这一点。

在初始化期间，运行时为系统中的每个设备创建CUDA上下文（有关CUDA上下文的更多详细信息，请参阅上下文）。 此上下文是此设备的主要上下文，它在应用程序的所有主机线程之间共享。 作为此上下文创建的一部分，设备代码在必要时即时编译（请参阅即时编译）并加载到设备内存中。 这一切都发生在幕后，运行时不会将主要上下文暴露给应用程序。

当主机线程调用cudaDeviceReset\(\)时，这会破坏主机线程当前操作的设备的主要上下文（即，设备选择中定义的当前设备）。 由此设备作为当前主机线程进行的下一个运行时函数调用将为此设备创建新的主要上下文。

#### Device Memory

如异构编程中所述，CUDA编程模型假定由主机和设备组成的系统，每个系统都有自己独立的内存。 内核在设备内存之外运行，因此运行时提供分配，释放和复制设备内存的功能，以及在主机内存和设备内存之间传输数据的功能。

设备存储器可以分配为线性存储器或CUDA数组。

CUDA数组是不透明的内存布局，针对纹理提取进行了优化。 它们在纹理和表面记忆中描述。

线性存储器存在于40位地址空间中的设备上，因此单独分配的实体可以通过指针相互引用，例如，在二叉树中。

线性内存通常使用cudaMalloc\(\)分配，并使用cudaFree\(\)释放，主机内存和设备内存之间的数据传输通常使用cudaMemcpy\(\)完成。在内核的向量加法代码示例中，向量需要从主机内存复制到设备内存:

```c
// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    ...
}
```

线性存储器也可以通过cudaMallocPitch\(\)和cudaMalloc3D\(\)分配。 建议将这些函数用于2D或3D数组的分配，因为它确保分配被适当填充以满足设备存储器访问中描述的对齐要求，从而确保在访问行地址或在2D数组与其他区域之间执行复制时的最佳性能 设备内存（使用cudaMemcpy2D\(\)和cudaMemcpy3D\(\)函数）。 返回的pitch \(or stride\)必须用于访问数组元素。 以下代码示例分配一个width x height的浮点值2D数组，并显示如何在设备代码中循环数组元素：

```c
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```

以下代码示例分配浮点值的width x height x depth 3D数组，并显示如何在设备代码中循环数组元素：

```c
// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
```

 参考手册列出了用于在使用cudaMalloc\(\)分配的线性内存，使用cudaMallocPitch\(\)或cudaMalloc3D\(\)分配的线性内存，CUDA数组以及为在全局或常量内存空间中声明的变量分配的内存之间复制内存的所有各种函数。

以下代码示例说明了通过运行时API访问全局变量的各种方法：

```c
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

cudaGetSymbolAddress\(\)用于检索指向为全局内存空间中声明的变量分配的内存的地址。分配的内存大小是通过cudaGetSymbolSize\(\)获得的。

#### Shared Memory

如变量内存空间说明符中所述，共享内存是使用\_\_shared\_\_内存空间说明符分配的。

共享内存预计比全局内存快得多，如线程层次结构中所述并在共享内存中详细说明。 因此，应该利用共享存储器访问替换全局存储器访问的任何机会，如以下矩阵乘法示例所示。

下面的代码示例是矩阵乘法的简单实现，它不利用共享内存。 每个线程读取A的一行和B的一列，并计算C的相应元素，如图9所示。因此A从全局存储器读取B.width次和B读取A.height次。

```c
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, Cd.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
```

Figure 9. Matrix Multiplication without Shared Memory

![](../../.gitbook/assets/image%20%28212%29.png)

以下代码示例是矩阵乘法的实现，它确实利用了共享内存。 在该实现中，每个线程块负责计算C的一个方形子矩阵Csub，并且块内的每个线程负责计算Csub的一个元素。 如图10所示，Csub等于两个长矩阵的乘积：具有与Csub相同的行索引的维度A（A.width，block\_size）的子矩阵，以及维度B的子矩阵 （block\_size，A.width）与Csub具有相同的列索引。 为了适应设备的资源，这两个长矩阵根据需要被分成维数block\_size的多个方形矩阵，并且Csub被计算为这些矩阵的乘积之和。 通过首先将两个对应的方形矩阵从全局存储器加载到共享存储器，一个线程加载一个元素，然后让每个线程计算乘积的一个元素。 每个线程将乘积的结果累积到一个寄存器中，一旦完成就将结果写入全局存储器。

通过这种区块方式计算，我们利用快速共享内存并节省大量全局内存带宽，因为A只从全局内存中读取（B.width / block\_size）次并且读取B（A.height / block\_size）次 。

前一代码示例中的Matrix类型使用stride字段进行扩充，因此可以使用相同类型有效地表示子矩阵。 \_\_device\_\_函数用于获取和设置元素并从矩阵构建任何子矩阵。

```c
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
```

Figure 10. Matrix Multiplication with Shared Memory

![](../../.gitbook/assets/image%20%28199%29.png)

## Hardware Implementation

## Performance Guidelines

## Appendix

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

## 



