# Programming Model

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

![](../../../.gitbook/assets/image%20%2834%29.png)

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

![](../../../.gitbook/assets/image%20%28234%29.png)

### Heterogeneous Programming

如图8所示，CUDA编程模型假设CUDA线程在物理上独立的设备上执行，该设备作为运行C程序的主机的协处理器运行。 例如，当内核在GPU上执行而其余的C程序在CPU上执行时就是这种情况。

CUDA编程模型还假设主机和设备都在DRAM中保持它们自己独立的存储空间，分别称为主机存储器和设备存储器。 因此，程序通过调用CUDA运行时（[在编程接口中描述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)）来管理内核可见的全局，常量和纹理内存空间。 这包括设备内存分配和释放以及主机和设备内存之间的数据传输。

Unified Memory提供托管内存以桥接主机和设备内存空间。 可以从系统中的所有CPU和GPU访问托管内存，作为具有公共地址空间的单个连贯内存映像。 此功能可以实现设备内存的超额预订，并且无需在主机和设备上显式镜像数据，从而大大简化了移植应用程序的任务。 有关统一内存的介绍，请参阅[统一内存编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)。

Figure 8. Heterogeneous Programming

![](../../../.gitbook/assets/image%20%2821%29.png)

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

