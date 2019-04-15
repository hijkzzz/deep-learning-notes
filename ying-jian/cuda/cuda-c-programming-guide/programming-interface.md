# Programming Interface

CUDA C为熟悉C编程语言的用户提供了一条简单的路径，可以轻松编写程序以供设备执行。

它由对C语言的最小扩展集和运行时库组成。

核心语言扩展已在编程模型中介绍。它们允许程序员将内核定义为C函数，并在每次调用函数时使用一些新语法来指定网格和块维度。可以在[C语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)中找到所有扩展的完整描述。必须使用nvcc编译包含其中某些扩展的任何源文件，如使用[NVCC编译中所述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)。

运行时在[编译工作流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-workflow)中介绍。 它提供在主机上执行的C函数，用于分配和释放设备内存，在主机内存和设备内存之间传输数据，管理具有多个设备的系统等。可以在CUDA参考手册中找到运行时的完整描述。

运行时构建在较低级别的C API（CUDA驱动程序API）之上，该API也可由应用程序访问。驱动程序API通过暴露较低级别的概念（例如CUDA上下文 - 设备的主机进程的模拟）和CUDA模块（设备的动态加载库的模拟）来提供额外的控制级别。大多数应用程序不使用驱动程序API，因为它们不需要这种额外的控制级别，并且在使用运行时时，上下文和模块管理是隐式的，从而产生更简洁的代码。 驱动程序API在[Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)中介绍，并在参考手册中有详细描述。

## Compilation with NVCC

可以使用称为PTX的CUDA指令集架构来编写内核，这在PTX参考手册中有所描述。 然而，使用诸如C的高级编程语言通常更有效。在这两种情况下，必须通过nvcc将内核编译成二进制代码以在设备上执行。

nvcc是一个编译器驱动程序，它简化了编译C或PTX代码的过程：它提供了简单而熟悉的命令行选项，并通过调用实现不同编译阶段的工具集来执行它们。 本节概述了nvcc工作流和命令选项。 完整的描述可以在nvcc用户手册中找到。

### Compilation Workflow

#### **Offline Compilation**

用nvcc编译的源文件可以包括主机代码（即，在主机上执行的代码）和设备代码（即，在设备上执行的代码）的混合。 nvcc的基本工作流程包括将设备代码与主机代码分离，然后：

* 将设备代码编译为汇编表（PTX代码）和/或二进制表（cubin对象），
* 通过必要的CUDA C运行时函数调用替换内核中引入的&lt;&lt;&lt; ... &gt;&gt;&gt;语法（并在执行配置中更详细地描述）来修改主机代码，以从PTX代码加载和启动每个编译的内核 和/或cubin对象。

修改后的主机代码既可以作为C代码输出，也可以使用其他工具进行编译，也可以通过让nvcc在上一个编译阶段调用主机编译器直接输出目标代码。

应用程序可以：

* 链接到已编译的主机代码（这是最常见的情况），
* 或者忽略修改后的主机代码（如果有）并使用CUDA驱动程序API（请参阅驱动程序API）来加载和执行PTX代码或cubin对象

#### **Just-in-Time Compilation**

应用程序在运行时加载的任何PTX代码都由设备驱动程序进一步编译为二进制代码。 这称为即时编译。 即时编译会增加应用程序加载时间，但允许应用程序受益于每个新设备驱动程序随附的任何新编译器改进。 它也是应用程序在编译应用程序时不存在的设备上运行的唯一方法，如应用程序兼容性中所述。

当设备驱动程序即时编译某些应用程序的某些PTX代码时，它会自动缓存生成的二进制代码的副本，以避免在后续应用程序调用中重复编译。 缓存（称为计算缓存）在升级设备驱动程序时自动失效，因此应用程序可以从设备驱动程序中内置的新实时编译器的改进中受益。

环境变量可用于控制即时编译，如CUDA环境变量中所述

### Binary Compatibility

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

## CUDA C Runtime

运行时在cudart库中实现，该库通过cudart.lib或libcudart.a静态链接到应用程序，或通过cudart.dll或libcudart.so动态链接。 需要cudart.dll和/或cudart.so进行动态链接的应用程序通常将它们作为应用程序安装包的一部分包含在内。 在链接到CUDA运行时的同一实例的组件之间传递CUDA运行时符号的地址是安全的。

所有入口点都以cuda为前缀。

正如 [Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming) 中提到的, CUDA编程模型假设一个系统由一个主机和一个设备组成，每个设备都有各自独立的内存。 [Device Memory](programming-interface.md#device-memory)概述了用于管理设备内存的运行时函数。

[Shared Memory](programming-interface.md#shared-memory) 说明了在线程层次结构中引入的共享内存的使用，以最大限度地提高性能。

[Page-Locked Host Memory](programming-interface.md#page-locked-host-memory) 引入页面锁定的主机内存，它需要将内核执行与主机和设备内存之间的数据传输重叠。

[Asynchronous Concurrent Execution](programming-interface.md#asynchronous-concurrent-execution) 描述了用于在系统中的各个级别启用异步并发执行的概念和API。

[Multi-Device System](programming-interface.md#multi-device-system) 显示了编程模型如何扩展到具有连接到同一主机的多个设备的系统。

[Error Checking](programming-interface.md#error-checking) 描述了如何正确检查运行时生成的错误。

[Call Stack](programming-interface.md#call-stack) 提到用于管理CUDA C调用堆栈的运行时函数。

[Texture and Surface Memory](programming-interface.md#texture-and-surface-memory)呈现纹理和表面存储空间，提供访问设备存储器的另一种方式; 它们还暴露了GPU纹理硬件的一个子集。

[Graphics Interoperability](programming-interface.md#graphics-interoperability) 介绍了运行时提供的各种功能，以便与两个主要的图形API，OpenGL和Direct3D进行互操作。

### Initialization

运行时没有明确的初始化函数; 它在第一次调用运行时函数时初始化（更具体地说，除了参考手册的设备和版本管理部分中的函数之外的任何函数）。 在计时运行时函数调用和从第一次调用运行时解释错误代码时，需要记住这一点。

在初始化期间，运行时为系统中的每个设备创建CUDA上下文（有关CUDA上下文的更多详细信息，请参阅上下文）。 此上下文是此设备的主要上下文，它在应用程序的所有主机线程之间共享。 作为此上下文创建的一部分，设备代码在必要时即时编译（请参阅即时编译）并加载到设备内存中。 这一切都发生在幕后，运行时不会将主要上下文暴露给应用程序。

当主机线程调用cudaDeviceReset\(\)时，这会破坏主机线程当前操作的设备的主要上下文（即，设备选择中定义的当前设备）。 由此设备作为当前主机线程进行的下一个运行时函数调用将为此设备创建新的主要上下文。

### Device Memory

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

### Shared Memory

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

![](../../../.gitbook/assets/image%20%28225%29.png)

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

![](../../../.gitbook/assets/image%20%28211%29.png)

### Page-Locked Host Memory

运行时提供允许使用页面锁定\(也称为固定\)主机内存\(与malloc\(\)分配的常规可分页主机内存相反\)的功能:

* cudaHostAlloc\(\)和cudaFreeHost\(\)分配和释放页面锁定的主机内存；
* cudaHostRegister\(\)页面锁定malloc\(\)分配的内存范围\(有关限制，请参见参考手册\)。

使用页面锁定主机内存有几个好处：

* 对于异步程序并发执行中提到的一些设备，页面锁定主机内存和设备内存之间的复制可以与内核执行同时执行。
* 在某些设备上，页面锁定的主机内存可以映射到设备的地址空间，无需像映射内存中详细描述的那样将其复制到设备内存或从设备内存中复制。
* 在具有前端总线的系统上，如果主机存储器被分配为页面锁定，则主机存储器和设备存储器之间的带宽更高，如果此外它被分配为写组合，则带宽甚至更高，如写组合存储器中所述。

然而，页面锁定的主机内存是一种稀缺资源，因此页面锁定内存中的分配将在可分页内存中分配之前很久就开始失败。此外，通过减少操作系统可用于分页的物理内存量，消耗过多的页锁定内存会降低整体系统性能。

简单的零拷贝CUDA示例附带了关于页面锁定内存api的详细文档。

#### **Portable Memory**

页面锁定内存块可以与系统中的任何设备结合使用\(有关多设备系统的更多详细信息，请参见多设备系统\)，但默认情况下，使用上述页面锁定内存的好处仅与分配该块时的当前设备结合使用\(并且所有设备共享相同的统一地址空间\(如果有\)，如统一虚拟地址空间中所述\)。为了使这些优点对所有设备都可用，需要通过将标志cudaHostAllocPortable传递给cudaHostAlloc\(\)来分配块，或者通过将标志cudaHostRegisterPortable传递给cudaHostRegister\(\)来进行页面锁定

#### **Write-Combining Memory**

默认情况下，页面锁定的主机内存被分配为可缓存的。可以通过将标志cudaHostAllocWriteCombined传递给cudaHostAlloc\(\)来选择性地将其分配为写组合。写组合内存释放了主机的L1和L2缓存资源，使更多缓存可供应用程序的其余部分使用。此外，在通过PCI Express总线传输期间，不会窥探写入组合内存，这可以将传输性能提高高达40%。

从主机的写入组合存储器读取速度非常慢，因此写入组合存储器通常应该用于主机仅写入的存储器。

#### **Mapped Memory**

还可以通过将标记cudaHostAllocMapped传递给cudaHostAlloc\(\)或将标记cudaHostRegisterMapped传递给cudaHostRegister\(\)将页面锁定的主机内存块映射到设备的地址空间中。因此，这样的块通常有两个地址:一个在由cudaHostAlloc\(\)或malloc\(\)返回的主机内存中，另一个在设备内存中，可以使用cudaHostGetDevicePointer\(\)检索，然后用于从内核中访问该块。唯一的例外是使用cudaHostAlloc\(\)分配的指针，以及统一虚拟地址空间中提到的主机和设备使用统一地址空间的情况。

直接从内核访问主机内存有几个优点：

* 不需要在设备存储器中分配块，也不需要在该块和主机存储器中的块之间复制数据；数据传输是根据内核的需要隐式执行的；
* 不需要使用流\(参见并发数据传输\)来将数据传输与内核执行重叠；源自内核的数据传输自动与内核执行重叠。

由于映射的页锁定内存在主机和设备之间共享，因此应用程序必须使用流或事件同步内存访问（请参阅异步并发执行）以避免任何潜在的读后写，写后读或写后 - 写入危险。

为了能够检索指向任何映射的页锁定内存的设备指针，必须在执行任何其他CUDA调用之前，通过使用cudaDeviceMapHost标志调用cudaSetDeviceFlags\(\)来启用页锁定内存映射。否则，cudaHostGetDevicePointer\(\)将返回一个错误。

如果设备不支持映射的页面锁定主机内存，cudaHostGetDevicePointer\(\)也会返回一个错误。应用程序可以通过检查canMapHostMemory设备属性\(请参见设备枚举\)来查询此功能，对于支持映射页面锁定主机内存的设备，该属性等于1。

请注意，从主机或其他设备的角度来看，在映射的页面锁定内存上运行的原子函数\(请参见原子函数\)不是原子函数。

另请注意，CUDA运行时要求将1字节，2字节，4字节和8字节自然对齐的加载和存储到从设备发起的主机内存，从主机和其他方面的角度保留为单个访问 设备。 在某些平台上，原子到内存可能会被硬件分解为单独的加载和存储操作。 这些组件加载和存储操作对保留自然对齐的访问具有相同的要求。 例如，CUDA运行时不支持PCI Express总线拓扑，其中PCI Express桥接器将8字节自然对齐写入分成设备和主机之间的两个4字节写入。

### Asynchronous Concurrent Execution

CUDA将以下操作公开为可以相互并发操作的独立任务:

* 主机上的计算；
* 设备上的计算；
* 内存从主机传输到设备；
* 内存从设备传输到主机；
* 给定设备内存中的内存传输；
* 设备之间的内存传输。

这些操作之间实现的并发级别取决于设备的功能集和计算能力，如下所述。

#### **Concurrent Execution between Host and Device**

通过异步库函数来促进并发主机执行，该异步库函数在设备完成所请求的任务之前将控制返回到主机线程。 使用异步调用时，许多设备操作可以排在一起，以便在适当的设备资源可用时由CUDA驱动程序执行。 这减轻了主机线程管理设备的大部分责任，使其可以自由地执行其他任务。 以下设备操作与主机异步：

内核启动；

* 单个设备内存中的内存副本；
* 64 KB或更小的内存块从主机到设备的内存拷贝；
* 由以异步结尾的函数执行的内存副本；
* 内存集函数调用。

程序员可以通过将CUDA\_LAUNCH\_BLOCKING环境变量设置为1来全局禁用系统上运行的所有CUDA应用程序的内核启动的异步性。此功能仅用于调试目的，不应用作使生产软件可靠运行的方法。

如果通过分析器（Nsight，Visual Profiler）收集硬件计数器，则内核启动是同步的，除非启用了并发内核分析。 如果异步内存副本涉及非页锁定的主机内存，则它们也将是同步的。

####  **Concurrent Kernel Execution**

一些计算能力2.x和更高的设备可以同时执行多个内核。 应用程序可以通过检查concurrentKernels设备属性（请参阅设备枚举）来查询此功能，对于支持它的设备，该属性等于1。

设备可以并发执行的最大内核启动次数取决于其计算能力，如表14所示。

来自一个CUDA上下文的内核不能与来自另一个CUDA上下文的内核同时执行。

使用许多纹理或大量本地内存的内核不太可能与其他内核并发执行。

#### **Overlap of Data Transfer and Kernel Execution**

一些设备可以在内核执行的同时执行与GPU之间的异步内存复制。应用程序可以通过检查asyncEngineCount设备属性\(请参见设备枚举\)来查询该功能，对于支持该功能的设备，该属性大于零。如果拷贝中包含主机内存，则必须对其进行页面锁定。

也可以在内核执行的同时执行设备内拷贝\(在支持并发内核设备属性的设备上\)和/或在设备之间拷贝\(对于支持异步注册属性的设备\)。使用标准内存复制功能启动设备内复制，目的地址和源地址位于同一设备上。

#### **Concurrent Data Transfers**

某些计算能力为2.x或更高的设备可能会与进出该设备的副本重叠。应用程序可以通过检查asyncEngineCount设备属性\(请参见设备枚举\)来查询此功能，对于支持它的设备，该属性等于2。为了重叠，传输中涉及的任何主机内存都必须是页面锁定的。

#### **Streams**

应用程序通过流管理上述并发操作。流是按顺序执行的命令序列\(可能由不同的主机线程发出\)。另一方面，不同的流可以彼此无序或同时执行它们的命令；这种行为没有保证，因此不应该依赖于正确性\(例如，内核间的通信是未定义的\)。

_Creation and Destruction_

通过创建流对象并将其指定为内核启动序列和主机&lt; - &gt;设备内存副本的流参数来定义流。 以下代码示例创建两个流，并在页锁定内存中分配float的数组hostPtr。

```c
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);
```

下面的代码示例将这些流中的每一个定义为从主机到设备的一个内存拷贝、一个内核启动和从设备到主机的一个内存拷贝的序列:

```c
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
```

每个流将其输入数组hostPtr的部分复制到设备内存中的数组inputDevPtr，通过调用MyKernel\(\)处理设备上的inputDevPtr，并将结果outputDevPtr复制回hostPtr的相同部分。 重叠行为描述了在此示例中流如何重叠，具体取决于设备的功能。 请注意，hostPtr必须指向页面锁定的主机内存才能发生任何重叠。

通过调用cudaStreamDestroy\(\)来释放流

```c
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

如果调用cudaStreamDestroy\(\)时设备仍在流中工作，那么一旦设备完成流中的所有工作，函数将立即返回，并且与流相关联的资源将自动释放。

_Default Stream_

内核启动和主机&lt; - &gt;设备内存副本未指定任何流参数，或者等效地将stream参数设置为零，将发布到默认流。 因此它们按顺序执行。

对于使用--default-stream每线程编译标志（或在包含CUDA标头（cuda.h和cuda\_runtime.h）之前定义CUDA\_API\_PER\_THREAD\_DEFAULT\_STREAM宏）编译的代码，默认流是常规流和每个主机线程 有自己的默认流。

对于使用--default-stream遗留编译标志编译的代码，默认流是一个称为NULL流的特殊流，每个设备都有一个用于所有主机线程的NULL流。 NULL流是特殊的，因为它会导致隐式同步，如隐式同步中所述。

对于未指定--default-stream编译标志而编译的代码，将--default-stream legacy视为默认值。

_Explicit Synchronization_

有多种方法可以显式地使流相互同步。

cudaDeviceSynchronize\(\)等待，直到所有主机线程的所有流中的所有前面的命令都完成。

cudaStreamSynchronize\(\)将一个流作为参数，并等待直到给定流中所有前面的命令完成。它可用于将主机与特定流同步，从而允许其他流继续在设备上执行。

cudaStreamWaitEvent\(\)将一个流和一个事件作为参数\(有关事件的描述，请参见事件\)，并使在调用cudaStreamWaitEvent\(\)之后添加到给定流中的所有命令延迟执行，直到给定事件完成。流可以是0，在这种情况下，调用cudaStreamWaitEvent\(\)后添加到任何流的所有命令都会等待事件。

cudaStreamQuery\(\)为应用程序提供了一种方法，可以知道流中所有前面的命令是否都已完成。

为了避免不必要的减速，所有这些同步功能通常最好用于计时目的，或者隔离失败的启动或内存拷贝。

_Implicit Synchronization_

如果主机线程在它们之间发出以下任何一个操作，则来自不同流的两个命令不能同时运行：

* 页面锁定主机内存分配，
* 设备内存分配，
* 设备内存集，
* 两个地址之间的内存复制到同一设备内存，
* 对NULL流的任何CUDA命令，在Compute Capability 3.x和Compute Capability 7.x中描述的L1 /共享内存配置之间的切换。

对于支持并发内核执行且计算能力为3.0或更低的设备，需要进行相关性检查以查看流内核启动是否完成的任何操作：

* 只有当CUDA上下文中任何流的所有先前内核启动的所有线程块都已开始执行时，才能开始执行;
* 阻止所有后来的内核启动从CUDA上下文中的任何流启动，直到检查内核启动完成为止。

需要依赖性检查的操作包括与正在检查的启动相同的流中的任何其他命令，以及对该流中cudaStreamQuery\(\)的任何调用。因此，应用程序应该遵循这些准则来提高并发内核执行的潜力:

* 所有独立操作应在相关操作之前发布
* 任何类型的同步都应该尽可能延迟

_Overlapping Behavior_

两个流之间的执行重叠量取决于向每个流发出命令的顺序，以及设备是否支持数据传输和内核执行的重叠\(参见数据传输和内核执行的重叠\)、并发内核执行\(参见并发内核执行\)和/或并发数据传输\(参见并发数据传输\)。

例如，在不支持并发数据传输的设备上，“创建”和“销毁”这两个代码示例流根本不重叠，因为从主机到设备的内存拷贝是在从设备到主机的内存拷贝发布到流\[0\]之后发布到流\[1\]的，所以它只能在从设备到主机发布到流\[0\]的内存拷贝完成后才开始。如果代码以下列方式重写\(假设设备支持数据传输和内核执行的重叠\)

```c
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
```

然后从发送到流\[1\]的主机到设备的内存复制与发送到流\[0\]的内核启动重叠。

在支持并发数据传输的设备上，创建和销毁的代码示例的两个流重叠：从主机到设备的内存副本发送到流\[1\]与从设备到主机的内存副本重叠发送到流\[0\] 甚至将内核启动发送到stream \[0\]（假设设备支持数据传输和内核执行的重叠）。 但是，对于计算能力为3.0或更低的设备，内核执行不可能重叠，因为在从设备到主机的内存复制发送到流\[0\]之后，第二次内核启动被发送到流\[1\]，因此它被阻塞直到 根据Implicit Synchronization，发送到stream \[0\]的第一个内核启动完成。 如果代码被重写如上，则内核执行重叠（假设设备支持并发内核执行），因为在从设备到主机的内存复制发送到流\[0\]之前，第二次内核启动被发送到流\[1\]。 但是，在这种情况下，从发送到流\[0\]的设备到主机的内存复制只与按照隐式同步发送到流\[1\]的内核启动的最后一个线程块重叠，后者只能代表总数的一小部分。 内核的执行时间。

_Callbacks_

 运行时提供了一种通过cudaStreamAddCallback\(\)在任何点将回调插入流的方法。回调是一个函数，一旦在回调完成之前向流发出所有命令，就会在主机上执行该函数。流0中的回调一旦在回调完成之前所有流中发出的所有前面的任务和命令都被执行。

下面的代码示例在将主机到设备的内存副本、内核启动和设备到主机的内存副本发布到两个流之后，将回调函数MyCallback添加到每个流中。在每个设备到主机的内存拷贝完成后，回调将在主机上开始执行。

```c
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *data){
    printf("Inside callback %d\n", (size_t)data);
}
...
for (size_t i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaStreamAddCallback(stream[i], MyCallback, (void*)i, 0);
}
        
```

 回调后在流中发出的命令\(或者如果回调是向流0发出的，则为向任何流发出的所有命令\)不会在回调完成之前开始执行。cudaStreamAddCallback\(\)的最后一个参数保留供将来使用。

回调不能\(直接或间接\)调用CUDA应用编程接口，因为如果回调导致死锁，它可能最终会等待自己。

_Stream Priorities_

可以在创建时使用cudaStreamCreateWithPriority\(\)指定流的相对优先级。允许的优先级范围，按\[最高优先级、最低优先级排序\]可以使用CudadeViceGetStreamPriorityRange\(\)函数获得。在运行时，当低优先级方案中的块完成时，高优先级流中的等待块被调度在它们的位置。

下面的代码示例获取当前设备允许的优先级范围，并创建具有最高和最低可用优先级的流。

```c
// get the range of stream priorities for this device
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
```

#### **Graphs**

 图表为CUDA中的工作提交提供了一个新模型。 图是一系列操作，例如内核启动，由依赖关系连接，与其执行分开定义。 这允许图表定义一次然后重复启动。 将图形的定义与其执行分开可以实现许多优化：首先，与流相比，CPU启动成本降低，因为大部分设置是事先完成的; 第二，向CUDA展示整个工作流程可以实现流的分段工作提交机制可能无法实现的优化。

要查看图表可能的优化，请考虑流中发生的情况：当您将内核放入流中时，主机驱动程序会执行一系列操作，以准备在GPU上执行内核。 设置和启动内核所必需的这些操作是必须为发出的每个内核支付的开销。 对于具有较短执行时间的GPU内核，此开销成本可能是整个端到端执行时间的重要部分。

使用图表的工作提交分为三个不同的阶段：定义，实例化和执行。

* 在定义阶段，程序会在图中创建操作的描述以及它们之间的依赖关系。
* 实例化获取图形模板的快照，验证它，并执行大部分工作的设置和初始化，目的是最小化启动时需要完成的工作。 生成的实例称为可执行图。
* 可执行图可以启动到流中，类似于任何其他CUDA工作。 它可以在不重复实例化的情况下启动任意次。

_Graph Structure_

操作在图形中形成一个节点。操作之间的依赖关系是边。这些依赖性限制了操作的执行顺序。

一旦操作所依赖的节点完成，就可以在任何时候调度该操作。时间安排由CUDA系统决定。

_Node Types_

 图形节点可以是以下之一:

* kernel
* 中央处理器功能调用
* 记忆拷贝
* memset函数
* 空节点
* 子图形:执行单独的嵌套图形。参见图11。

Figure 11. Child Graph Example

![](../../../.gitbook/assets/image%20%28158%29.png)

_Creating a Graph Using Graph APIs_

可以通过两种机制创建图形：显式API和流捕获。 以下是创建和执行下图的示例。

Figure 12. Creating a Graph Using Graph APIs Example

![](../../../.gitbook/assets/image%20%28124%29.png)

```c
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified 
// at node creation. 
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

// Now set up dependencies on each node
cudaGraphAddDependencies(graph, &a, &b, 1);     // A->B
cudaGraphAddDependencies(graph, &a, &c, 1);     // A->C
cudaGraphAddDependencies(graph, &b, &d, 1);     // B->D
cudaGraphAddDependencies(graph, &c, &d, 1);     // C->D
```

Creating a Graph Using Stream Capture

流捕获提供了一种从现有基于流的API创建图的机制。 通过调用cudaStreamBeginCapture\(\)和cudaStreamEndCapture\(\)，可以将包含现有代码的一部分代码启动到流中，包括现有代码。 见下文。

```c
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);
```

对cudaStreamBeginCapture\(\)的调用会将流置于捕获模式。 捕获流时，启动到流中的工作不会排队执行。 它被附加到逐步建立的内部图形。 然后通过调用cudaStreamEndCapture\(\)返回该图，该结果也结束了流的捕获模式。 通过流捕获主动构建的图被称为捕获图。

除了cudaStreamLegacy\(“NULL stream”\)之外，可以在任何CUDA流上使用流捕获。 请注意，它可以在cudaStreamPerThread上使用。 如果程序正在使用遗留流，则可以将流0重新定义为每线程流而不进行功能改变。 请参阅默认流。

是否正在捕获流可以使用cudaStreamIsCapturing\(\)查询。

Cross-stream Dependencies and Events

流捕获可以处理用cudaEventRecord\(\)和cudaStreamWaitEvent\(\)表示的跨流依赖关系，前提是被等待的事件被记录到同一个捕获图中。

当在处于捕获模式的流中记录事件时，将导致捕获事件。捕获的事件表示捕获图中的一组节点。

当一个捕获的事件被一个流等待时，如果它还没有被捕获，它将把这个流放在捕获模式中，并且流中的下一项将对捕获事件中的节点具有额外的依赖关系。然后将这两个流捕获到。

```c
// stream1 is the origin stream
cudaStreamBeginCapture(stream1);

kernel_A<<< ..., stream1 >>>(...);

// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);

// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernel_D<<< ..., stream1 >>>(...);

// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);

// stream1 and stream2 no longer in capture mode   
```

上面代码返回的图形如图12所示。

注意：当流从捕获模式中取出时，流中的下一个未捕获项（如果有）仍将依赖于最近的先前未捕获项，尽管中间项已被删除。

Prohibited and Unhandled Operations

同步或查询正在捕获的流或捕获的事件的执行状态是无效的，因为它们不代表计划执行的项目。当任何关联流处于捕获模式时，查询包含活动流捕获的更宽句柄的执行状态或同步该句柄也是无效的，例如设备或上下文句柄。

当捕获同一个上下文中的任何流，并且该流不是用cudaStreamNonBlocking创建的时，任何对旧流的尝试都是无效的。这是因为遗留流句柄始终包含这些其他流；排队到传统流会对被捕获的流产生依赖性，并且查询或同步它会查询或同步被捕获的流。

因此，在这种情况下调用同步APIs也是无效的。同步APIs，如cudaMemcpy\(\)，将工作排入遗留流，并在返回之前对其进行同步。

注意:一般来说，当依赖关系将捕获的东西与未捕获的东西连接起来，而不是排队执行时，CUDA更喜欢返回错误而不是忽略依赖关系。将流置于捕获模式或脱离捕获模式时会出现异常；这切断了模式转换前后添加到流中的项目之间的依赖关系。

通过等待正在被捕获的流中与不同于事件的捕获图相关联的捕获事件来合并两个单独的捕获图是无效的。等待正在捕获的流中的未捕获事件是无效的。

图形中目前不支持将异步操作排入流中的少量APIs，如果用正在捕获的流调用这些APIs，例如cudaStreamAttachMemAsync\(\)。

Invalidation

当在流捕获期间尝试无效操作时，任何关联的捕获图都将无效。当捕获图无效时，进一步使用任何正在捕获的流或与该图相关联的捕获事件都是无效的，并将返回错误，直到流捕获以cudaStreamEndCapture\(\)结束。该调用将使相关流脱离捕获模式，但也会返回错误值和零图值。

_Using Graph APIs_

cudaGraph\_t对象不是线程安全的。 用户有责任确保多个线程不会同时访问相同的cudaGraph\_t。

cudaGraphExec\_t无法与自身同时运行。 cudaGraphExec\_t的启动将在同一可执行图的先前启动之后进行。

 图形执行是在流中完成的，以便与其他异步工作进行排序。然而，该流仅用于排序;它不限制图的内部并行性，也不影响图节点的执行位置。

请参阅图API。

#### Events

运行时还通过让应用程序异步记录程序中任意点的事件并在这些事件完成时进行查询，提供了一种密切监视设备进度以及执行精确定时的方法。当事件之前的所有任务\(或者可选地，给定流中的所有命令\)完成时，该事件就完成了。流0中的事件在所有流中的所有先前任务和命令完成后完成。

_Creation and Destruction_

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
```

```c
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

_Elapsed Time_

```c
cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>
               (outputDev + i * size, inputDev + i * size, size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
```

### Multi-Device System

主机系统可以有多个设备。 以下代码示例演示如何枚举这些设备，查询其属性以及确定启用CUDA的设备的数量。

```c
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```

_Device Selection_

主机线程可以通过调用cudaSetDevice\(\)来设置它在任何时候操作的设备。在当前设置的设备上进行设备内存分配和内核启动；与当前设置的设备相关联地创建流和事件。如果没有调用cudaSetDevice\(\)，则当前设备是设备0。

下面的代码示例说明了设置当前设备如何影响内存分配和内核执行。

```c
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // Set device 0 as current
float* p0;
cudaMalloc(&p0, size);       // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
cudaSetDevice(1);            // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);       // Allocate memory on device 1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
```

_Stream and Event Behavior_

 如果将内核启动发布到与当前设备无关的流，则内核启动将失败，如以下代码示例所示。

```c
cudaSetDevice(0);               // Set device 0 as current
cudaStream_t s0;
cudaStreamCreate(&s0);          // Create stream s0 on device 0
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0
cudaSetDevice(1);               // Set device 1 as current
cudaStream_t s1;
cudaStreamCreate(&s1);          // Create stream s1 on device 1
MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1

// This kernel launch will fail:
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0
```

即使内存拷贝被发布到与当前设备不相关联的流，它也将成功。

如果输入事件和输入流与不同的设备相关联，cudaEventRecord\(\)将失败。

如果两个输入事件与不同的设备相关联，cudaEventElapsedTime\(\)将失败。

即使输入事件与不同于当前设备的设备相关联，cudaEventSynchronize\(\)和cudaEventQuery\(\)也会成功。

即使输入流和输入事件与不同的设备相关联，cudaStreamWaitEvent\(\)也会成功。cudaStreamWaitEvent\(\)因此可以用来使多个设备相互同步。

每个设备都有自己的默认流\(请参见默认流\)，因此向设备的默认流发出的命令可能会无序执行，或者与向任何其他设备的默认流发出的命令同时执行。

_Peer-to-Peer Memory Access_

当应用程序作为64位进程运行时，特斯拉系列中计算能力为2.0或更高的设备可以寻址彼此的存储器\(即，在一个设备上执行的内核可以取消引用指向另一个设备的存储器的指针\)。如果这两个设备的cudaDeviceCanAccessPeer\(\)返回true，则这种对等内存访问功能在这两个设备之间受支持。

必须通过调用cudaDeviceEnablePeerAccess\(\)在两个设备之间启用对等内存访问，如下面的代码示例所示。在未启用NVSwitch的系统上，每个设备最多可以支持八个系统范围的对等连接。

两个设备都使用统一地址空间\(请参见统一虚拟地址空间\)，因此可以使用相同的指针来寻址来自两个设备的内存，如下面的代码示例所示_。_

```c
cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
cudaSetDevice(1);                   // Set device 1 as current
cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                    // with device 0

// Launch kernel on device 1
// This kernel launch can access memory on device 0 at address p0
MyKernel<<<1000, 128>>>(p0);
```

IOMMU on Linux

 仅在Linux上，CUDA和显示驱动程序不支持启用IOMMU的裸机PCIe对等内存拷贝。但是，CUDA和显示驱动程序确实通过虚拟机通道支持IOMMU。因此，当在本机裸机系统上运行时，Linux上的用户应该禁用IOMMU。应启用IOMMU，并将VFIO驱动程序用作虚拟机的PCIe通道。

在Windows系统上，上述限制并不存在。

另请参见在64位平台上分配DMA缓冲区。

_Peer-to-Peer Memory Copy_

 存储器拷贝可以在两个不同设备的存储器之间执行。

当两个设备都使用统一地址空间时\(请参见统一虚拟地址空间\)，这是使用设备内存中提到的常规内存复制功能来完成的。

否则，这将使用cudaMemcpyPeer\(\)、cudaMemcpyPeerAsync\(\)、cudaMemcpy3DPeer\(\)或cudaMemcpy3DPeerAsync\(\)来完成，如下面的代码示例所示

```c
cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0
cudaSetDevice(1);                   // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);              // Allocate memory on device 1
cudaSetDevice(0);                   // Set device 0 as current
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
cudaSetDevice(1);                   // Set device 1 as current
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1
```

两个不同设备的存储器之间的副本（在隐式NULL流中）：

* 直到先前发给任一设备的所有命令都完成后才会启动
* 在复制到任一设备之后发出的任何命令（请参阅异步并发执行）之前，运行完成。

与流的正常行为一致，两个设备的存储器之间的异步复制可能与另一个流中的副本或内核重叠。

请注意，如果通过对等内存访问中所述的cudaDeviceEnablePeerAccess\(\)在两个设备之间启用了对等访问，则这两个设备之间的对等内存复制不再需要通过主机进行， 因此更快。

### Unified Virtual Address Space

当应用程序作为64位进程运行时，主机和所有计算能力为2.0或更高的设备使用一个地址空间。通过CUDA应用编程接口调用进行的所有主机内存分配和支持设备上的所有设备内存分配都在此虚拟地址范围内。因此:

* 通过CUDA分配的主机上或使用统一地址空间的任何设备上的任何内存的位置，都可以通过使用cudaPointerGetAttributes\(\)的指针值来确定。
* 当复制到使用统一地址空间的任何设备的内存中或从其中复制时，cudaMemcpy \*\(的cudaMemcpyKind参数\)可以设置为cudaMemcpyDefault，以根据指针确定位置。这也适用于没有通过CUDA分配的主机指针，只要当前设备使用统一寻址。
* 通过cudaHostAlloc\(\)的分配在使用统一地址空间的所有设备上是自动可移植的\(参见可移植内存\)，并且cudaHostAlloc\(\)返回的指针可以直接从这些设备上运行的内核中使用\(即，不需要通过cudaHostGetDevicePointer\(\)获得设备指针，如映射内存中所述。

应用程序可以通过检查统一寻址设备属性\(参见设备枚举\)是否等于1来查询统一地址空间是否用于特定设备

### Interprocess Communication

主机线程创建的任何设备内存指针或事件句柄都可以被同一进程中的任何其他线程直接引用。但是，它在此进程之外无效，因此不能被属于不同进程的线程直接引用。

为了跨进程共享设备内存指针和事件，应用程序必须使用行程间通讯应用编程接口，这在参考手册中有详细描述。IPC应用编程接口仅支持Linux上的64位进程和计算能力为2.0或更高的设备。请注意，cudaMallocManaged分配不支持IPC应用编程接口。

使用此应用程序接口，应用程序可以使用cudaIpcGetMemHandle\(\)获取给定设备内存指针的IPC句柄，并使用标准的IPC机制\(例如进程间共享内存或文件\)将其传递给另一个进程，然后使用cudaIpcOpenMemHandle\(\)从IPC句柄中检索设备指针，该指针是另一个进程中的有效指针。事件句柄可以使用类似的入口点共享。

使用仪表板组合仪表应用编程接口的一个例子是，单个主进程生成一批输入数据，使得数据可用于多个从进程，而不需要再生或复制。

使用CUDA IPC的应用程序应该使用相同的CUDA驱动程序和运行时进行编译、链接和运行。

注:图睿设备不支持CUDA IPC调用。

### Error Checking

所有运行时函数都返回一个错误代码，但是对于异步函数\(参见异步程序并发执行\)，该错误代码不可能报告设备上可能发生的任何异步错误，因为该函数在设备完成任务之前返回；错误代码仅报告在执行任务之前发生在主机上的错误，通常与参数验证有关；如果发生异步错误，它将由一些随后不相关的运行时函数调用来报告。

因此，在某个异步函数调用之后检查异步错误的唯一方法是在调用之后通过调用cudaDeviceSynchronize\(\)\(或通过使用异步程序并发执行中描述的任何其他同步机制\)并检查cudaDeviceSynchronize\(\)返回的错误代码来进行同步。

运行时为初始化为cudaSuccess的每个主机线程维护一个错误变量，并在每次出错时被错误代码覆盖\(无论是参数验证错误还是异步错误\)。cudaPeekAtLastError\(\)返回此变量。cudaGetLastError\(\)返回此变量，并将其重置为cudaSuccess。

内核启动不会返回任何错误代码，因此必须在内核启动后立即调用cudaPeekAtLastError\(\)或cudaGetLastError\(\)，以检索任何启动前错误。为了确保cudaPeekAtLastError\(\)或cudaGetLastError\(\)返回的任何错误都不是来自内核启动之前的调用，必须确保运行时错误变量在内核启动之前设置为cudaSuccess，例如，在内核启动之前调用cudaGetLastError\(\)。内核启动是异步的，因此为了检查异步错误，应用程序必须在内核启动和对cudaPeekAtLastError\(\)或cudaGetLastError\(\)的调用之间进行同步。

请注意，cudaErrorNotReady\(可能由cudaStreamQuery\(\)和cudaEventQuery\(\)返回\)不被视为错误，因此不会由cudaPeekAtLastError\(\)或cudaGetLastError\(\)报告。

### Call Stack

在计算能力为2.x及更高的设备上，可以使用cudaDeviceGetLimit\(\)查询调用堆栈的大小，并使用cudaDeviceSetLimit\(\)进行设置。

当调用堆栈溢出时，如果应用程序通过CUDA调试器\(cuda-gdb，Nsight\)运行，内核调用将失败，并出现堆栈溢出错误，否则为。

### Texture and Surface Memory

CUDA支持纹理硬件的一个子集，GPU用于图形访问纹理和表面存储器。 从纹理或表面存储器而不是全局存储器读取数据可以具有若干性能优势，如设备存储器访问中所述。

有两种不同的API可以访问纹理和表面内存：

* 所有设备都支持的纹理参考API，
* 纹理对象API仅在计算能力3.x的设备上受支持。

纹理参考API具有纹理对象API不具有的限制。 它们在Texture Reference API中提到。

#### Texture Memory

使用纹理函数中描述的设备函数从内核读取纹理内存。 读取调用这些函数之一的纹理的过程称为纹理提取。 每个纹理提取指定一个称为纹理对象API的纹理对象的参数或纹理参考API的纹理参考。

纹理对象或纹理参考指定：

* 纹理，是获取的纹理内存块。 纹理对象在运行时创建，并且在创建纹理对象时指定纹理，如Texture Object API中所述。 纹理引用是在编译时创建的，纹理是在运行时指定的，方法是通过运行时函数将纹理引用绑定到纹理，如Texture Reference API中所述; 几个不同的纹理引用可能绑定到相同的纹理或内存中重叠的纹理。 纹理可以是线性存储器的任何区域或CUDA数组（在CUDA数组中描述）。
* 它的维数指定纹理是使用一个纹理坐标作为一维数组，使用两个纹理坐标作为二维数组，还是使用三个纹理坐标作为三维数组数组。数组的元素被称为纹理元素，是纹理元素的简称。纹理宽度、高度和深度指的是每个维度中数组的大小。表14根据设备的计算能力列出了最大纹理宽度、高度和深度。
* 纹理元素的类型，仅限于基本整数和单精度浮点类型，以及从基本整数和单精度浮点类型派生的char、short、int、long、long、float、double中定义的任何1、2和4分量向量类型。
* 读取模式，等于cudaReadModeNormalizedFloat或cudaReadModeElementType。 如果它是cudaReadModeNormalizedFloat并且texel的类型是16位或8位整数类型，则纹理提取返回的值实际上作为浮点类型返回，并且整数类型的整个范围映射到\[0.0 ，1.0表示无符号整数类型，\[-1.0,1.0\]表示有符号整数类型; 例如，值为0xff的无符号8位纹理元素读取为1.如果是cudaReadModeElementType，则不执行转换。
* 纹理坐标是否标准化。 默认情况下，使用\[0，N-1\]范围内的浮点坐标引用纹理（通过纹理函数的函数），其中N是对应于坐标的维度中纹理的大小。 例如，对于x和y维度，尺寸为64x32的纹理将分别用\[0,63\]和\[0,31\]范围内的坐标引用。 归一化纹理坐标导致坐标在\[0.0,1.0-1 / N\]范围内而不是\[0，N-1\]中指定，因此相同的64x32纹理将通过范围\[0,1-\]中的归一化坐标来寻址 x和y维度均为1 / N\]。 如果纹理坐标优选独立于纹理大小，则标准化纹理坐标自然适合某些应用程序的要求。
* 寻址模式。使用超出范围的坐标调用B.8节的设备功能是有效的。寻址模式定义了在这种情况下会发生什么。默认寻址模式是将坐标箝位到有效范围:\[0，N\)用于非归一化坐标，\[0.0，1.0\)用于归一化坐标。如果改为指定边框模式，纹理坐标超出范围的纹理提取将返回零。对于归一化坐标，还可以使用环绕模式和镜像模式。使用环绕模式时，每个坐标x转换为frac\(x\)=x floor\(x\)，其中floor\(x\)是不大于x的最大整数。使用镜像模式时，如果floor\(x\)为偶数，每个坐标x转换为frac\(x\)，如果floor\(x\)为奇数，则转换为1-frac\(x\)。寻址模式被指定为大小为三的阵列，其第一、第二和第三元素分别为第一、第二和第三纹理坐标指定寻址模式；寻址模式是cudaAddressModeBorder、cudaAddressModeClamp、cudaAddressModeWrap和cudaAddressModeMirrorcudaAddressModeWrap和cudaAddressModeMirror仅支持归一化纹理坐标。
* 过滤模式，指定在获取纹理时如何根据输入的纹理坐标计算返回值。线性纹理过滤只能对被配置为返回浮点数据的纹理进行。它在相邻纹理元素之间执行低精度插值。启用时，读取纹理提取位置周围的纹理元素，并基于纹理坐标落在纹理元素之间的位置对纹理提取的返回值进行插值。一维纹理执行简单的线性插值，二维纹理执行双线性插值，三维纹理执行三线性插补处理插值。纹理提取提供了纹理提取的更多细节。过滤模式等于cudaFilterModePoint或CudaFilterModeLink。如果是cudaFilterModePoint，返回值是纹理坐标最接近输入纹理坐标的纹理元素。如果是cudaFilterModeLinear，则返回值是纹理坐标最接近输入纹理坐标的两个\(对于一维纹理\)、四个\(对于二维纹理\)或八个\(对于三维纹理\)纹理元素的线性插值。cudaFilterModeLinear仅对浮点类型的返回值有效

[Texture Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api)介绍纹理对象API。

[Texture Reference API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-reference-api) 介绍纹理对象API。

[16-Bit Floating-Point Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#sixteen-bit-floating-point-textures)  解释了如何处理16位浮点纹理。

纹理也可以分层，如 [Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#layered-textures)所述。

[Cubemap Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cubemap-textures) and [Cubemap Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cubemap-layered-textures) 描述一种特殊类型的纹理，即立方体贴图纹理。

[Texture Gather](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-gather) 描述了一种特殊的纹理提取，纹理聚集。

_Texture Object API_

纹理对象是使用cudaCreateTextureObject\(\)从指定纹理的struct cudaResourceDesc类型的资源描述和这样定义的纹理描述创建的:

```c
struct cudaTextureDesc
{
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode  filterMode;
    enum cudaTextureReadMode    readMode;
    int                         sRGB;
    int                         normalizedCoords;
    unsigned int                maxAnisotropy;
    enum cudaTextureFilterMode  mipmapFilterMode;
    float                       mipmapLevelBias;
    float                       minMipmapLevelClamp;
    float                       maxMipmapLevelClamp;
};
```

_Texture Reference API_

纹理引用的一些属性是不可变的，必须在编译时知道; 在声明纹理参考时指定它们。 纹理引用在文件范围内声明为纹理类型的变量：

```c
texture<DataType, Type, ReadMode> texRef;
```

 其中:

* DataType指定纹理元素的类型；
* Type指定纹理引用的类型，对于一维、二维或三维纹理，类型分别等于cudaTextureType1D、cudaTextureType2D或cudaTextureType3D，对于一维或二维分层纹理，类型分别等于CuDatextureType1d分层或CuDatextureType2d分层；类型是一个可选参数，默认为cudaTextureType1D
* ReadMode指定读取模式；这是一个可选参数，默认为cudaReadModeElementType。

 纹理引用只能声明为静态全局变量，不能作为参数传递给函数。

纹理引用的其他属性是可变的，可以在运行时通过主机运行时更改。 如参考手册中所述，运行时API具有低级C风格接口和高级C ++风格接口。 纹理类型在高级API中定义为从低级API中定义的textureReference类型公开派生的结构，如下所示：

```c
struct textureReference {
    int                          normalized;
    enum cudaTextureFilterMode   filterMode;
    enum cudaTextureAddressMode  addressMode[3];
    struct cudaChannelFormatDesc channelDesc;
    int                          sRGB;
    unsigned int                 maxAnisotropy;
    enum cudaTextureFilterMode   mipmapFilterMode;
    float                        mipmapLevelBias;
    float                        minMipmapLevelClamp;
    float                        maxMipmapLevelClamp;
}
```

* normalized 指定纹理坐标是否规格化；
* filterMode 指定过滤模式；
* addressMode 指定寻址模式；
* channelDesc 描述纹理元素的格式；它必须匹配纹理引用声明的DataType参数；信道类型如下:

```c
struct cudaChannelFormatDesc {
  int x, y, z, w;
  enum cudaChannelFormatKind f;
};
```

其中x，y，z和w等于返回值的每个分量的位数，f是

* cudaChannelFormatKindSigned if these components are of signed integer type,
* cudaChannelFormatKindUnsigned if they are of unsigned integer type,
* cudaChannelFormatKindFloat if they are of floating point type.

有关sRGB，maxAnisotropy，mipmapFilterMode，mipmapLevelBias，minMipmapLevelClamp和maxMipmapLevelClamp的参考手册，请参阅参考手册。

normalized，addressMode和filterMode可以在主机代码中直接修改。

在内核可以使用纹理引用从纹理内存中读取之前，必须使用cudaBindTexture\(\)或cudaBindTexture2D\(\)为线性内存将纹理引用绑定到纹理，或者为CUDA数组使用cudaBindTextureToArray\(\)。 cudaUnbindTexture\(\)用于取消绑定纹理引用。 一旦纹理引用被解除绑定，它就可以安全地反弹到另一个数组，即使使用先前绑定纹理的内核尚未完成。 建议使用cudaMallocPitch\(\)在线性存储器中分配二维纹理，并使用cudaMallocPitch\(\)返回的pitch作为cudaBindTexture2D\(\)的输入参数

以下代码示例将2D纹理引用绑定到devPtr指向的线性内存：

* Using the low-level API:

```c
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
textureReference* texRefPtr;
cudaGetTextureReference(&texRefPtr, &texRef);
cudaChannelFormatDesc channelDesc =
                             cudaCreateChannelDesc<float>();
size_t offset;
cudaBindTexture2D(&offset, texRefPtr, devPtr, &channelDesc,
                  width, height, pitch);
```

* Using the high-level API:

```c
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
cudaChannelFormatDesc channelDesc =
                             cudaCreateChannelDesc<float>();
size_t offset;
cudaBindTexture2D(&offset, texRef, devPtr, channelDesc,
                  width, height, pitch);
```

以下代码示例将2D纹理引用绑定到CUDA数组cuArray

* Using the low-level API:

```c
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
textureReference* texRefPtr;
cudaGetTextureReference(&texRefPtr, &texRef);
cudaChannelFormatDesc channelDesc;
cudaGetChannelDesc(&channelDesc, cuArray);
cudaBindTextureToArray(texRef, cuArray, &channelDesc);
```

* Using the high-level API:

```c
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
cudaBindTextureToArray(texRef, cuArray);
```

将纹理绑定到纹理引用时指定的格式必须与声明纹理引用时指定的参数匹配; 否则，纹理提取的结果是未定义的。

如表14所示，可以绑定到内核的纹理数量有限制。

以下代码示例将一些简单的转换内核应用于纹理。

```c
/ 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Simple transformation kernel
__global__ void transformKernel(float* output,
                                int width, int height,
                                float theta) 
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    // Transform coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;


    // Read from texture and write to global memory
    output[y * width + x] = tex2D(texRef, tu, tv);
}

// Host code
int main()
{
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Copy to device memory some data located at address h_data
    // in host memory 
    cudaMemcpyToArray(cuArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // Set texture reference parameters
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode     = cudaFilterModeLinear;
    texRef.normalized     = true;

    // Bind the array to the texture reference
    cudaBindTextureToArray(texRef, cuArray, channelDesc);

    // Allocate result of transformation in device memory
    float* output;
    cudaMalloc(&output, width * height * sizeof(float));

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    transformKernel<<<dimGrid, dimBlock>>>(output, width, height,
                                           angle);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);

    return 0;
}
```

_16-Bit Floating-Point Textures_

CUDA数组支持的16位浮点或半格式与IEEE 754-2008 binary2格式相同。

CUDA C不支持匹配的数据类型，但提供了通过无符号短类型转换为32位浮点格式的内部函数：\_\_ fllo2half\_rn（float）和\_\_half2float（unsigned short）。 这些功能仅在设备代码中受支持。 例如，可以在OpenEXR库中找到主机代码的等效函数。

在执行任何过滤之前，在纹理获取期间，16位浮点组件被提升为32位浮点数。

可以通过调用cudaCreateChannelDescHalf \*（）函数之一来创建16位浮点格式的通道描述。

_Layered Textures_

一维或二维分层纹理（在Direct3D中也称为纹理数组和OpenGL中的数组纹理）是由一系列图层组成的纹理，所有图层都是具有相同维度，大小和数据类型的常规纹理 。

使用整数索引和浮点纹理坐标来寻址一维分层纹理; 索引表示序列中的层，坐标表示该层内的纹素。 使用整数索引和两个浮点纹理坐标来寻址二维分层纹理; 索引表示序列中的一个层，坐标表示该层内的一个纹素。

通过使用cudaArrayLayered标志调用cudaMalloc3DArray（）（一维分层纹理的高度为零），分层纹理只能是CUDA数组。

使用tex1DLayered\(\)，tex1DLayered\(\)，tex2DLayered\(\)和tex2DLayered\(\)中描述的设备函数获取分层纹理。 纹理过滤（请参见纹理提取）仅在图层内完成，而不是跨层。

分层纹理仅在计算能力2.0及更高版本的设备上受支持。

 _Cubemap Textures_

 立方体贴图纹理是一种特殊类型的二维分层纹理，有六层表示立方体的面:

层的宽度等于它的高度。

立方体贴图使用三个纹理坐标x、y和z来寻址，这三个坐标被解释为从立方体的中心发出的方向向量，指向立方体的一个面和对应于该面的层内的纹理元素。更具体地，通过具有最大幅度m的坐标来选择面，并且使用坐标\(s/m+1\)/2和\(t/m+1\)/2来寻址相应的层，其中s和t在表1中定义。

![](../../../.gitbook/assets/image%20%2854%29.png)

通过使用cudaArrayCubemap标志调用cudaMalloc3DArray\(\)，分层纹理只能是CUDA数组。

使用texCubemap\(\)和texCubemap\(\)中描述的设备函数获取立方体贴图纹理。

仅在计算能力2.0及更高版本的设备上支持立方体贴图纹理。

_Cubemap Layered Textures_

立方体贴图分层纹理是一种分层纹理，其图层是相同维度的立方体贴图。

使用整数索引和三个浮点纹理坐标来寻址立方体贴图分层纹理; 索引表示序列中的立方体贴图，坐标表示该立方体贴图中的纹理像素。

通过使用cudaArrayLayered和cudaArrayCubemap标志调用cudaMalloc3DArray\(\)，分层纹理只能是CUDA数组。

使用texCubemapLayered\(\)和texCubemapLayered\(\)中描述的设备函数获取立方体贴图分层纹理。 纹理过滤（请参见纹理提取）仅在图层内完成，而不是跨层。

仅在计算能力2.0及更高版本的设备上支持Cubemap分层纹理。

_Texture Gather_

纹理聚集是一种特殊的纹理提取，仅适用于二维纹理。 它由tex2Dgather\(\)函数执行，该函数与tex2D\(\)具有相同的参数，另外还有一个等于0,1,2或3的comp参数（参见tex2Dgather\(\)和tex2Dgather\(\)）。 它返回四个32位数字，这些数字对应于在常规纹理提取期间用于双线性滤波的四个纹素的每一个的分量comp的值。 例如，如果这些纹素具有值（253,20,31,255），（250,25,29,254），（249,16,37,253），（251,22,30,250），以及 comp为2，tex2Dgather\(\)返回（31,29,37,30）。

请注意，纹理坐标仅使用8位小数精度计算。 因此tex2Dgather（）可能会返回意外的结果，因为tex2D（）将使用1.0作为其权重之一（α或β，请参见线性过滤）。 例如，x纹理坐标为2.49805：xB = x-0.5 = 1.99805，但xB的小数部分以8位定点格式存储。 由于0.99805接近256.f / 256.f而不是255.f / 256.f，因此xB的值为2.因此，在这种情况下，tex2Dgather（）将返回x中的索引2和3，而不是索引 1和2。

纹理聚集仅支持使用cudaArrayTextureGather标志创建的CUDA数组，其宽度和高度小于表14中为纹理聚集指定的最大值，该值小于常规纹理提取。

纹理聚集仅在计算能力2.0及更高版本的设备上受支持。

#### Surface Memory

对于计算能力为2.0或更高的设备，可以使用surface函数中描述的函数，通过surface对象或surface引用，读取和写入使用cudaArraySurfaceLoadStore标志创建的CUDA数组\(在立方体贴图曲面中描述\)。

表14列出了取决于器件计算能力的最大表面宽度、高度和深度。

_Surface Object API_

 表面对象是使用cudaCreateSurfaceObject\(\)从结构类型cudaResourceDesc的资源描述中创建的。

下面的代码示例将一些简单的转换内核应用于纹理。

```c
// Simple copy kernel
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
                           cudaSurfaceObject_t outputSurfObj,
                           int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // Read from input surface
        surf2Dread(&data,  inputSurfObj, x * 4, y);
        // Write to output surface
        surf2Dwrite(data, outputSurfObj, x * 4, y);
    }
}

// Host code
int main()
{
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(8, 8, 8, 8,
                                   cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);

    // Copy to device memory some data located at address h_data
    // in host memory 
    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outputSurfObj = 0;
    cudaCreateSurfaceObject(&outputSurfObj, &resDesc);

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    copyKernel<<<dimGrid, dimBlock>>>(inputSurfObj,
                                      outputSurfObj,
                                      width, height);


    // Destroy surface objects
    cudaDestroySurfaceObject(inputSurfObj);
    cudaDestroySurfaceObject(outputSurfObj);

    // Free device memory
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);

    return 0;
}
```

_Surface Reference API_

表面引用在文件范围内声明为surface类型的变量

```c
surface<void, Type> surfRef;
```

其中类型指定曲面参照的类型，并等于cudaSurfaceType1D、cudaSurfaceType2D、cudaSurfaceType3D、cudaSurfaceTypeCubemap、cudaSurfaceType1D分层、cudaSurfaceType2D分层或CuDaSurfacetypeCubemap分层；类型是一个可选参数，默认为cudaSurfaceType1D。表面引用只能声明为静态全局变量，不能作为参数传递给函数。

在内核可以使用表面引用来访问CUDA数组之前，必须使用cudaBindSurfaceToArray\(\)将表面引用绑定到CUDA数组。

以下代码示例将表面引用绑定到CUDA数组cuArray:

* Using the low-level API:

```c
surface<void, cudaSurfaceType2D> surfRef;
surfaceReference* surfRefPtr;
cudaGetSurfaceReference(&surfRefPtr, "surfRef");
cudaChannelFormatDesc channelDesc;
cudaGetChannelDesc(&channelDesc, cuArray);
cudaBindSurfaceToArray(surfRef, cuArray, &channelDesc);
```

* Using the high-level API:

```c
surface<void, cudaSurfaceType2D> surfRef;
cudaBindSurfaceToArray(surfRef, cuArray);
```

必须使用匹配维度和类型的表面函数以及匹配维度的表面参考来读取和写入CUDA数组; 否则，读取和写入CUDA数组的结果是不确定的。

与纹理内存不同，表面内存使用字节寻址。 这意味着用于通过纹理函数访问纹理元素的x坐标需要乘以元素的字节大小，以通过表面函数访问同一元素。 例如，通过texRef使用tex1d\(texRef，x\)读取绑定到纹理参考texRef和表面参考surfRef的一维浮点CUDA数组的纹理坐标x处的元素，但是surf1Dread\(surfRef，4 \* x\) ）通过surfRef。 类似地，通过texRef使用tex2d\(texRef，x，y\)访问绑定到纹理参考texRef和表面参考surfRef的二维浮点CUDA数组的纹理坐标x和y处的元素，但是surf2Dread\(\)urfRef， 4 \* x，y）通过surfRef（y坐标的字节偏移在内部根据CUDA阵列的基线间距计算）。

以下代码示例将一些简单的转换内核应用于纹理。

```c
// 2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;
            
// Simple copy kernel
__global__ void copyKernel(int width, int height) 
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // Read from input surface
        surf2Dread(&data,  inputSurfRef, x * 4, y);
        // Write to output surface
        surf2Dwrite(data, outputSurfRef, x * 4, y);
    }
}

// Host code
int main()
{
    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(8, 8, 8, 8,
                                   cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);

    // Copy to device memory some data located at address h_data
    // in host memory 
    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // Bind the arrays to the surface references
    cudaBindSurfaceToArray(inputSurfRef, cuInputArray);
    cudaBindSurfaceToArray(outputSurfRef, cuOutputArray);

    // Invoke kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    copyKernel<<<dimGrid, dimBlock>>>(width, height);


    // Free device memory
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);

    return 0;
```

_Cubemap Surfaces_

使用thesurfCubemapread\(\)和surfCubemapwrite\(\)（surfCubemapread和surfCubemapwrite）作为二维分层表面访问立方体贴图表面，即使用表示面的整数索引和两个浮点纹理坐标来寻址对应于此面的图层内的纹理元素 。 面如表1所示。

_Cubemap Layered Surfaces_

使用surfCubemapLayeredread\(\)和surfCubemapLayeredwrite\(\)（surfCubemapLayeredread\(\)和surfCubemapLayeredwrite\(\)）作为二维分层表面访问Cubemap分层曲面，即使用表示其中一个立方体贴图和两个浮点纹理的面的整数索引 寻找对应于该面部的层内的纹理元素的坐标。 面如表1所示排序，因此索引（（2 \* 6）+ 3）例如访问第三个立方体贴图的第四个面。

#### CUDA Arrays

CUDA数组是不透明的内存布局，针对纹理提取进行了优化。 它们是一维的，二维的或三维的，由元素组成，每个元素都有1,2或4个分量，可以是有符号或无符号的8位，16位或32位整数，16位浮点数， 或32位浮点数。 CUDA数组只能由内核通过纹理获取访问，如纹理存储器中所述，或表面读取和写入，如表面存储器中所述。

#### Read/Write Coherency

纹理和表面存储器被缓存（参见设备存储器访问）并且在相同的内核调用中，缓存在全局存储器写入和表面存储器写入方面不保持一致，因此任何纹理提取或表面读取到已经存在的地址 通过全局写入或表面写入写入同一内​​核调用返回未定义的数据。 换句话说，只有当先前的内核调用或内存副本更新了此内存位置时，线程才能安全地读取某些纹理或表面内存位置，但如果先前已由相同线程或其他线程更新 内核调用。

### Graphics Interoperability

来自OpenGL和Direct3D的一些资源可以被映射到CUDA的地址空间，以使CUDA能够读取由OpenGL或Direct3D写入的数据，或者使CUDA能够写入供OpenGL或Direct3D消费的数据。

在使用OpenGL互操作性和Direct3D互操作性中提到的函数映射资源之前，必须将资源注册到CUDA。这些函数返回一个指向结构化CUDA图形资源的指针。注册资源的开销可能很高，因此通常每个资源只调用一次。使用CUDA图形资源注销CUDA图形资源注销资源\(\)。每个打算使用该资源的CUDA上下文都需要单独注册。

一旦资源注册到CUDA，就可以使用CUDA映射资源\(\)和CUDA映射取消映射资源\(\)。可以调用CUDA驱动程序来指定CUDA驱动程序可以用来优化资源管理的使用提示\(只读、只读\)。

内核可以使用cudagraphicsresourceGetMappedpointer\(\)为缓冲区返回的设备内存地址和cudagraphicsSubreSourceGetPartarray\(\)为CUDA数组返回的设备内存地址来读取或写入映射的资源。

在映射资源时，通过OpenGL、Direct3D或另一CUDA上下文访问资源会产生未定义的结果。OpenGL互操作性和Direct3D互操作性给出了每个图形应用编程接口和一些代码示例的细节。SLI互操作性给出了系统何时处于SLI模式的细节。

#### OpenGL Interoperability

可以映射到CUDA地址空间的OpenGL资源是OpenGL缓冲区、纹理和渲染缓冲区对象。

缓冲区对象是使用cudaGraphicsGLRegisterBuffer\(\)注册的。在CUDA中，它表现为一个设备指针，因此可以由内核或通过cudaMemcpy\(\)调用读写。

纹理或渲染缓冲区对象使用cudaGraphicsGLRegisterImage\(\)注册。在CUDA中，它显示为CUDA数组。内核可以通过将数组绑定到纹理或表面引用来读取数组。如果资源已经用CudagraphicsRegisterflagsSuffacealoadStore标志注册，它们也可以通过表面写函数写入。数组也可以通过cudaMemcpy2D\(\)调用读写。cudaGraphicsGLRegisterImage\(\)支持所有具有1、2或4个组件和内部类型浮点\(如GL\_RGBA\_FLOAT32\)、归一化整数\(如GL\_RGBA8、GL \_ INTENSITY16\)和非归一化整数\(如GL\_RGBA8UI\)的纹理格式\(请注意，由于非归一化整数格式需要OpenGL 3.0，它们只能由着色器编写，不能由固定函数管道编写\)。

共享资源的OpenGL上下文必须对进行任何OpenGL互操作性应用编程接口调用的主机线程是最新的。

请注意:当OpenGL纹理变成无绑定时\(例如，通过使用GlgetTerreHandle \*/GlgetimageHandle \* APIs请求图像或纹理句柄\)，它不能在CUDA中注册。在请求图像或纹理句柄之前，应用程序需要为interop注册纹理。

下面的代码示例使用内核动态修改存储在顶点缓冲对象中的顶点的2D宽×高网格:

```c
GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

int main()
{
    // Initialize OpenGL and GLUT for device 0
    // and make the OpenGL context current
    ...
    glutDisplayFunc(display);

    // Explicitly set device 0
    cudaSetDevice(0);

    // Create buffer object and register it with CUDA
    glGenBuffers(1, &positionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    unsigned int size = width * height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA,
                                 positionsVBO,
                                 cudaGraphicsMapFlagsWriteDiscard);

    // Launch rendering loop
    glutMainLoop();

    ...
}

void display()
{
    // Map buffer object for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,  
                                         positionsVBO_CUDA));

    // Execute kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);

    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);

    // Swap buffers
    glutSwapBuffers();
    glutPostRedisplay();
}
```

```c
void deleteVBO()
{
    cudaGraphicsUnregisterResource(positionsVBO_CUDA);
    glDeleteBuffers(1, &positionsVBO);
}

__global__ void createVertices(float4* positions, float time,
                               unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // Write positions
    positions[y * width + x] = make_float4(u, w, v, 1.0f);
}
```

在Windows和Quadro GPU上，cudaWGLGetDevice\(\)可用于检索与wglEnumGpusNV\(\)返回的句柄相关联的CUDA设备。 Quadro GPU在多GPU配置中提供比GeForce和Tesla GPU更高的性能OpenGL互操作性，其中OpenGL渲染在Quadro GPU上执行，CUDA计算在系统中的其他GPU上执行。

#### Direct3D Interoperability

Direct3D 9Ex、Direct3D 10和Direct3D 11支持Direct3D互操作性。

CUDA上下文只能与满足以下标准的Direct3D设备进行互操作:创建Direct3D 9Ex设备时，设备类型必须设置为D3DDEVTYPE\_HAL，行为标志必须设置为D3DCREATE \_硬件\_ VERTEXPROCESSING标志；Direct3D 10和Direct3D 11设备必须使用设置为D3D驱动程序类型硬件的驱动程序类型创建。

可以映射到CUDA地址空间的直接3D资源是直接3D缓冲区、纹理和表面。这些资源是使用cudagraphicsd3d 9 RegisterReSource\(\)、cudagraphicsd3d 10 RegisterReSource\(\)和cudagraphicsd3d 11 RegisterReSource\(\)注册的。

下面的代码示例使用内核动态修改存储在顶点缓冲对象中的顶点的2D宽×高网格

```c
ID3D11Device* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
ID3D11Buffer* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;

int main()
{
    int dev;
    // Get a CUDA-enabled adapter
    IDXGIFactory* factory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    IDXGIAdapter* adapter = 0;
    for (unsigned int i = 0; !adapter; ++i) {
        if (FAILED(factory->EnumAdapters(i, &adapter))
            break;
        if (cudaD3D11GetDevice(&dev, adapter) == cudaSuccess)
            break;
        adapter->Release();
    }
    factory->Release();

    // Create swap chain and device
    ...
    sFnPtr_D3D11CreateDeviceAndSwapChain(adapter, 
                                         D3D11_DRIVER_TYPE_HARDWARE,
                                         0, 
                                         D3D11_CREATE_DEVICE_DEBUG,
                                         featureLevels, 3,
                                         D3D11_SDK_VERSION, 
                                         &swapChainDesc, &swapChain,
                                         &device,
                                         &featureLevel,
                                         &deviceContext);
    adapter->Release();

    // Use the same device
    cudaSetDevice(dev);

    // Create vertex buffer and register it with CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.Usage          = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth      = size;
    bufferDesc.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags      = 0;
    device->CreateBuffer(&bufferDesc, 0, &positionsVB);
    cudaGraphicsD3D11RegisterResource(&positionsVB_CUDA,
                                      positionsVB,
                                      cudaGraphicsRegisterFlagsNone);
    cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                    cudaGraphicsMapFlagsWriteDiscard);

    // Launch rendering loop
    while (...) {
        ...
        Render();
        ...
    }
    ...
}
```

```c
void Render()
{
    // Map vertex buffer for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,  
                                         positionsVB_CUDA));

    // Execute kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // Unmap vertex buffer
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);

    // Draw and present
    ...
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

    __global__ void createVertices(float4* positions, float time,
                          unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

// Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // Calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // Write positions
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
}
```

#### SLI Interoperability

 在具有多个GPU的系统中，所有支持CUDA的GPU都可以通过CUDA驱动程序和运行时作为单独的设备访问。 但是，当系统处于SLI模式时，有如下所述的特殊注意事项。

首先，在一个GPU上的一个CUDA设备中的分配将消耗作为Direct3D或OpenGL设备的SLI配置的一部分的其他GPU上的存储器。 因此，分配可能会比预期的更早失败。

其次，应用程序应创建多个CUDA上下文，SLI配置中的每个GPU都有一个。 虽然这不是严格的要求，但它避免了设备之间不必要的数据传输。 应用程序可以使用用于Direct3D的cudaD3D \[9 \| 10 \| 11\] GetDevices（）和用于OpenGL调用的cudaGLGetDevices（）来识别正在执行当前渲染的设备的CUDA设备句柄 和下一帧。 鉴于此信息，当deviceList参数设置为cudaD3D时，应用程序通常会选择适当的设备并将Direct3D或OpenGL资源映射到cudaD3D \[9 \| 10 \| 11\] GetDevices（）或cudaGLGetDevices（）返回的CUDA设备\[9 \| 10\] \| 11\] DeviceListCurrentFrame或cudaGLDeviceListCurrentFrame。

请注意，从cudaGraphicsD9D \[9 \| 10 \| 11\] RegisterResource和cudaGraphicsGLRegister \[Buffer \| Image\]返回的资源必须仅在注册发生的设备上使用。 因此，在SLI配置上，当在不同的CUDA设备上计算不同帧的数据时，必须分别为每个设备注册资源。

有关CUDA运行时如何与Direct3D和OpenGL进行互操作的详细信息，请参阅Direct3D互操作性和OpenGL互操作性。

## Versioning and Compatibility

开发CUDA应用程序时，开发人员应该关心两个版本号:描述计算设备的一般规范和特性的计算能力\(参见计算能力\)和描述驱动程序API和运行时支持的特性的CUDA驱动程序API的版本。

驱动程序API的版本在驱动程序头文件中定义为CUDA\_VERSION。它允许开发人员检查他们的应用程序是否需要比当前安装的设备驱动程序更新的设备驱动程序。这很重要，因为驱动程序接口是向后兼容的，这意味着针对驱动程序接口的特定版本编译的应用程序、插件和库\(包括运行时\)将继续在后续的设备驱动程序版本中工作，如图13所示。驱动程序应用程序接口不向前兼容，这意味着针对驱动程序应用程序接口的特定版本编译的应用程序、插件和库\(包括运行时\)将无法在设备驱动程序的早期版本上运行。

请注意，受支持版本的混合和匹配存在限制:

* 由于一个系统上一次只能安装一个版本的CUDA驱动程序，因此安装的驱动程序的版本必须与该系统上必须运行的任何应用程序、插件或库的最大驱动程序应用编程接口版本相同或更高。
* 应用程序使用的所有插件和库必须使用相同版本的CUDA运行时，除非它们静态链接到运行时，在这种情况下，运行时的多个版本可以共存于同一进程空间中。请注意，如果使用nvcc来链接应用程序，默认情况下将使用CUDA运行时库的静态版本，并且所有CUDA工具包库都与CUDA运行时静态链接。
* 应用程序使用的所有插件和库必须使用使用运行时的任何库的相同版本\(例如cuFFT、cuBLAS，...\)除非静态链接到那些库。

Figure 13. The Driver API Is Backward but Not Forward Compatible

![](../../../.gitbook/assets/image%20%28176%29.png)

## Compute Modes

在运行Windows Server 2008及更高版本或Linux的Tesla解决方案中，可以使用NVIDIA的系统管理界面（nvidia-smi）以三种以下模式之一设置系统中的任何设备，该系统是作为驱动程序的一部分分发的工具：

* 默认计算模式：多个主机线程可以使用该设备（通过在此设备上调用cudaSetDevice\(\)，使用运行时API，或者在使用驱动程序API时同时使当前与设备关联的上下文）。
* 独占进程计算模式：系统中的所有进程只能在设备上创建一个CUDA上下文。 在创建该上下文的进程中，上下文可以是所需数量的线程。
* 禁止的计算模式：无法在设备上创建CUDA上下文。

这尤其意味着，如果设备0处于禁止模式或独占进程模式，并且被另一个进程使用，则使用运行时应用编程接口而不显式调用cudaSetDevice\(\)的主机线程可能与设备0以外的设备相关联。cudaSetValidDevices\(\)可用于从设备的优先级列表中设置设备。

还要注意，对于采用帕斯卡体系结构的设备\(主要版本号为6及更高版本的计算能力\)，存在对计算抢占的支持。这允许在指令级粒度抢占计算任务，而不是像现有的麦克斯韦和开普勒图形处理器体系结构那样抢占线程块粒度，其好处是可以防止具有长时间运行内核的应用程序垄断系统或超时。但是，将会有与计算抢占相关联的关联切换开销，该开销会在支持的设备上自动启用。具有支持的属性CudadeVattrComputePreepEnsupported的单个属性查询函数cudaDeviceGetAttribute\(\)可用于确定正在使用的设备是否支持计算抢占。希望避免与不同进程相关联的关联切换开销的用户可以通过选择独占进程模式来确保在GPU上只有一个进程是活动的。

应用程序可以通过检查计算模式设备属性来查询设备的计算模式\(请参见设备枚举\)。

## Mode Switches

 具有显示输出的GPU将一些DRAM存储器专用于所谓的主表面，其用于刷新其输出被用户查看的显示设备。 当用户通过更改显示器的分辨率或位深度（使用NVIDIA控制面板或Windows上的显示控制面板）启动显示模式切换时，主表面所需的内存量会发生变化。 例如，如果用户将显示分辨率从1280x1024x32位更改为1600x1200x32位，则系统必须将7.68 MB专用于主表面而不是5.24 MB。 （启用了抗锯齿运行的全屏图形应用程序可能需要更多的主表面显示内存。）在Windows上，可能启动显示模式切换的其他事件包括启动全屏DirectX应用程序，按Alt + Tab键到任务 切换到全屏DirectX应用程序，或按Ctrl + Alt + Del锁定计算机。

如果模式开关增加了主表面所需的内存量，则系统可能不得不蚕食专用于CUDA应用程序的内存分配。 因此，模式切换会导致对CUDA运行时的任何调用失败并返回无效的上下文错误。

## Tesla Compute Cluster Mode for Windows

使用NVIDIA的系统管理界面（nvidia-smi），可以将Windows设备驱动程序置于TCC（特斯拉计算集群）模式，用于特斯拉和Quadro系列计算能力2.0及更高版本的设备。

此模式具有以下主要优点：

它可以在具有非NVIDIA集成显卡的集群节点中使用这些GPU;

它通过远程桌面提供这些GPU，直接和通过依赖远程桌面的集群管理系统;

它使这些GPU可用于作为Windows服务运行的应用程序（即，在会话0中）。

但是，TCC模式删除了对任何图形功能的支持。

