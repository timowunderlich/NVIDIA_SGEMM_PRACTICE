![](images/head.png)

![](https://img.shields.io/badge/build-passing-brightgreen) ![](https://img.shields.io/badge/ubuntu-18.04-blue) ![](https://img.shields.io/badge/cuda-10.2-blue) ![](https://img.shields.io/badge/nvidia-RTX3090-blue) ![](https://img.shields.io/badge/cmake-3.21-blue)

# Overview

Step-by-step optimization of matrix multiplication performance for NVIDIA GPUs using CUDA programming:

| Kernel | Description | GFLOPS | Custom kernel/CUBLAS (%) |
| ------ | ----------- | ------ | ------------------------ |
| CUBLAS | Official library function | 14448.69 | Baseline |
| kernel_1 | Naive implementation | 2262.168 | 15.65657 |
| kernel_2 | Shared memory caching | 4216.536 | 29.18283 |
| kernel_3 | One-dimensional Thread Tile parallel optimization | 7809.629 | 54.05078 |
| kernel_4 | Two-dimensional Thread Tile parallel optimization | 12251.3 | 84.79179 |
| kernel_5 | Register caching | 12177.95 | 84.28412 |
| kernel_6 | FLOAT4 vector memory access | 13161.49 | 91.09125 |
| kernel_7 | Double buffering prefetch | 13634.98 | 94.36832 |

> NVIDIA GeForce RTX 3090, matrix size 5120

# Configuration

- Compilation using `gcc 7.5.0` under Ubuntu 18.04.5 LTS
- NVIDIA CUDA version: `CUDA 10.2`

# Directory Structure

```
NVIDIA_SGEMM_PRACTICE                                   # Root directory
    ├── images                                          # Image results
    │     ├── describe_kernel_1.png  
    │     ├── describe_kernel_x.png
    │     └── kernel_x_vs_y.png
    ├── test                                            # Test results
    │     ├── test_kernel_0.txt 
    │     ├── test_kernel_1.txt 
    │     └── test_kernel_x.txt 
    └── src                                             # Source files
    │    ├── kernel
    │    │  ├── kernel_1.cuh                            # Declarations and definitions
    │    │  ├── kernel_2.cuh
    │    │  └── kernel_x.cuh
    │    ├── kernel.cuh
    │    ├── utils.cuh                                  # Helper functions
    │    └── utils.cu
    ├── plot.py                                         # Plot based on test results
    ├── run.sh                                          # Run compiled executable
    ├── sgemm.cu                                        # Main program
    └── CMakeLists.txt                                  # Compilation related
```

# Running

1. Configure NVCC compilation parameters
   > Modify `set(CUDA_NVCC_FLAGS -arch=compute_70;-code=compute_70)` in CMakeLists.txt
2. Configure maximum matrix calculation size
   > Modify `size_len` in `sgemm.cu:16`. It's recommended to set it to 16 for initial runs, as too large sizes may cause power overload and host restart.
3. Compile
   `cd build && cmake .. && make`
4. Run run.sh to calculate the efficiency of each kernel function, with results saved in the test directory.
5. Plot efficiency line chart

   > `python plot.py 0 1` will plot the comparison of CUBLAS and kernel_1 calculation efficiency.

# Step-by-Step Optimization

## kernel 1

**Naive basic matrix multiplication implementation**

Each logical thread corresponds to each element of matrix C, with each thread responsible for calculating one element in C.

![](./images/describe_kernel_1.png)

```cpp
__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // Global x
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // Global y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx]; // Two global memory accesses and one FMA (fused multiply-add)
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}
```

![](./images/kernel_culas_vs_1.png)

The unoptimized matrix multiplication performance is less than 1/10 of CUBLAS. Specific analysis:

- Compute-to-memory access ratio: Each iteration requires one FMA operation and two global memory reads, resulting in a compute-to-memory ratio of 1/2.
- Memory access: Accessing global memory, calculating each element of matrix C requires accessing `2K` single-precision floating-point numbers, completing all calculations requires `2*K*M*N`.

Global memory access has high latency (hundreds of cycles), and the same elements are repeatedly read (elements in the same row of C share elements in the same row of A, elements in the same column of C share elements in the same column of B). Additionally, the low compute-to-memory ratio cannot effectively hide memory access latency. Therefore, memory access latency and compute-to-memory ratio are the reasons for kernel 1's low efficiency.

## kernel 2

**Using shared memory cache to reduce global memory access and latency**

Memory access latency comes from both high global memory latency and repeated global memory access. Shared memory is on-chip memory with lower access latency (tens of cycles). Using shared memory as a cache can reduce access latency.

![](./images/describe_kernel_2.png)

> BM and BN represent the height and width of the block tile, BK represents the stride of global memory to be cached, meaning a block calculation needs to cache K/BK times.

Shared memory caches global memory A tile and B tile, completes FMA calculations for all elements in C block, continuously slides the cache area, and updates the block.

```cpp
/*
dim3 blockDim(1024);
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
mysgemm_v2<32><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/

template<const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    // Allocate shared memory space
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move to current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {
        // Cache A_tile and B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // Synchronize all threads to complete caching
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA calculation needs to read cached data, synchronize before new round of caching to ensure all threads complete calculations
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}
```

![](./images/kernel_1_vs_2.png)

- Memory access: Each block needs to read `(K/BK)*(BM*BK+BK*BN)` single-precision floating-point numbers from global memory. There are `(M/BM)*(N/BN)` blocks in the entire C, so completing all element calculations in C requires reading `(M/BM)*(N/BN)*(K/BK)*(BM*BK+BK*BN)` single-precision floating-point numbers.

Kernel 1 was limited by global memory access latency and repeated access. Before optimization, the global memory access was `2*K*M*N`. After shared memory caching optimization, the memory access is reduced to `1/2*(1/BN)*(1/BM)` of the original. When `BN=BM=32`, memory access is reduced to 1/32. Additionally, shared memory access latency is much lower than global memory, so calculation efficiency has improved to some extent.

## kernel 3

**One-dimensional thread tile optimization**

We know that by increasing the block size (BM, BN) values, we can further reduce global memory access. Therefore, BM and BN are increased from 32 to 64.

> **Can global memory access be reduced indefinitely by increasing block size?**
>
> No. On one hand, if the block matrix size is too large and the number of blocks is reduced, this will cause a large number of SM (Streaming Multiprocessor) to be idle and wasted. On the other hand, increasing BN and BM requires more shared memory, and the more shared memory occupied by a single thread, the fewer active thread bundles, which is not conducive to hiding instruction latency.

Therefore, while increasing BM and BN values, to reduce shared memory usage, BK value is reduced to 8.

> When increasing block size, particular attention should be paid to shared memory consumption, limiting shared memory size and the number of threads in the block to avoid kernel function startup failure due to insufficient resources.

![](./images/describe_kernel_3_1.png)

On the other hand, shared memory caching reduced global memory access and FMA multiply-accumulate access latency, but the compute-to-memory ratio has not improved. Each iteration calculation requires two memory access instructions and one calculation instruction. Therefore, thread tile is introduced, meaning a thread is responsible for calculating multiple elements in the block. TM and TN represent the height and width of the thread tile.

![](./images/describe_kernel_3_2.png)

```cpp
/*
dim3 blockDim(512);
dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
mysgemm_v3<64, 64, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
*/


template<const int BM,
        const int BN,
        const int BK,
        const int TM>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = BM * BN / TM; // One thread is responsible for calculating TM elements in the block

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move to current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    /*
    The current thread is responsible for moving global memory from row a_tile_row, column a_tile_col to shared memory row a_tile_row, column a_tile_col
    a_tile_stride indicates threads in block can move a_tile_stride rows to shared memory;

    If BM=64,BK=8,thread_num=512, then a_tile_stride=64,a_tile_stride=BM, indicating each thread can complete the required element movement in one round;
    If BM=128,BK=8,thread_num=512, then a_tile_stride=64, indicating each thread needs two rounds to complete the required element movement;
    */
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM + 1] = {0.}; // Each thread is responsible for TM elements, so TM registers need to be allocated to save the accumulated values, with an extra register for caching
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            tmp[TM] = Bs[tx + i * BN]; // An extra register to avoid repeatedly reading Bs[tx + i * BN] from shared memory
            #pragma unroll  // Loop unrolling to increase instruction parallelism
            for (int j = 0; j < TM; j++) {
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; j++) {
        C[(ty + j) * N + tx] = alpha * tmp[j] + beta * C[(ty + j) * N + tx];
    }
}
```

![](./images/kernel_2_vs_3.png)

This example optimizes from two aspects:

- Global memory access: Compared to the initial version, by caching a `64*64` block size, memory access is reduced to 1/64.
- Compute-to-memory ratio: Introducing thread tile, using a single thread to be responsible for calculating multiple elements, increasing the compute-to-memory ratio. When TM=8, executing 8 shared memory As access instructions and 1 shared memory Bs access instruction can execute 8 calculation instructions. Compared to the initial version's compute-to-memory ratio of 1:2, it improves to 8:9, effectively hiding memory access latency.

Through these two aspects of optimization, matrix multiplication calculation efficiency significantly improves by nearly double.

## kernel 4

**Two-dimensional thread tile optimization**

Set the thread tile to be two-dimensional, meaning one thread is responsible for calculating a small block of elements, further increasing the block size and reducing global memory access.

> Increasing thread tile size can calculate larger block sizes with the same or fewer number of threads.

More importantly, a single thread is responsible for calculating more C element areas, which can increase the degree of instruction-level parallelism.

> Why can it improve instruction parallelism?
>
> The more instructions a single thread processes, the longer the pipeline level. Since a single thread pipeline can process multiple instructions in parallel, although a single instruction execution becomes slower, the number of instructions processed per unit time increases, improving throughput and hiding instruction latency. Instruction-level concurrency has more advantages than thread-level concurrency.

![](./images/describe_kernel_4.png)

Set one thread to be responsible for calculating elements in an 8×8 area, that is, thread tile=8×8, TM=8, TN=8.

```cpp
// BM=BN=128, BK=8, TM=TN=8, shared memory size 128*8
dim3 blockDim(256);
dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
mysgemm_v4<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;  // 128*8/256=4, all threads need to move 4 rounds to move a 128*8 area from global memory to shared memory

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

// Each thread is responsible for TM*TN elements, so TM*TN registers need to be allocated to save the accumulated values
float tmp[TM][TN] = {0.}; 

// A single thread loops through TM, TN to complete the multiply-accumulate of elements within the thread tile
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
}
```

Global memory access: Compared to the version without shared memory caching, global memory access is reduced to `1/2*(1/BM+1/BN)=1/128`, significantly reducing memory access.

![](./images/kernel_3_vs_4.png)

Actual testing found that, compared to one-dimensional thread tile, two-dimensional thread tile further reduces global memory access and improves compute-to-memory ratio, significantly doubling matrix multiplication efficiency.

## kernel 5

**Register caching shared memory**

![](./images/describe_kernel_5.png)

As can be seen from the code below, when a single thread calculates thread tile element multiply-accumulate, shared memory is repeatedly accessed.

```cpp
for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
        tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];  // In the inner loop, As[(ty + j) * BK + i] is accessed TN times repeatedly
}
```

Shared memory greatly reduces access latency compared to global memory, but shared memory latency (tens of cycles) is still large compared to computation latency (a few cycles). Therefore, registers are used to cache shared memory As and Bs to avoid repeated shared memory access.

```cpp
float a_frag[TM] = {0.};
float b_frag[TN] = {0.};

for (int i = 0; i < BK; i++) {
    for (int j = 0; j < TM; j++) {
        a_frag[j] = As[(ty + j) * BK + i];     // Use a_frag register array to cache the As shared memory data needed by thread tile
    }
    for (int l = 0; l < TN; l++) {
        b_frag[l] = Bs[tx + l + i * BN];       // Use b_frag register array to cache the Bs shared memory data needed by thread tile
    }
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            tmp[j][l] += a_frag[j] * b_frag[l];
    }
}
```

When TM=TN=8, after register caching, each thread tile needs to execute 8 As shared memory access instructions and 8 Bs shared memory access instructions, and can perform 8×8=64 calculation instructions. The compute-to-memory ratio improves from the initial version's 1/2 to 64:16, effectively hiding memory access latency.

![](./images/kernel_4_vs_5.png)

Actual testing found that register caching did not result in a significant change in performance. The reason might be that the current performance bottleneck is not repeated shared memory access.

## kernel 6

**FLOAT4 vector memory instruction optimization**

- Calculation instruction: GPU performs calculations in 4-dimensional vectors as the basic unit. A float4 vector composed of 4 floating-point numbers is the most basic type of GPU. Using the GPU to calculate two float4 vectors is the same as calculating two integers or two floating-point numbers, requiring only one instruction.
- Memory instruction: Compared to issuing a single instruction to generate separate memory transactions to obtain the same number of bytes, fewer memory transactions are required through vector memory instructions, reducing contention for the memory controller. On the other hand, using vector loading requires fewer index calculations per byte.

![](./images/describe_kernel_6.png)

For example, BM=128, BK=8, thread count is 256. If each thread fetches 1 floating-point number each time, each thread needs to consume 4 memory instructions to move global memory to shared memory. If using float4 vector memory instructions, each thread can move 4 floating-point numbers each time, and each thread only needs to execute one memory instruction to complete the movement.

Key code example:

```cpp
#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float ldg_a_reg[4 * ldg_a_num] = {0.}; // Each thread moves ldg_a_num rounds, register caches ldg_a_num float4 elements, used to transpose As matrix

//  Shared memory caches global memory
for (int i = 0; i < BM; i += a_tile_stride) {
    int ldg_index = i / a_tile_stride * 4;  // The ldg_index round
    FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
            FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
    // As is stored transposed, where ldg_a_reg is used as an intermediate cache, the purpose is to read by FLOAT4 when reading
    As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
    As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
    As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
    As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
}

for (int i = 0; i < BK; i += b_tile_stride) {
    FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
        FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // No need to transpose
}


// Register caches shared memory
// ty, tx are the positions of the top-left element of the thread tile corresponding to the current thread in the block
#pragma unroll
for (int m = 0; m < TM; m += 4) {
    FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]); // Offset to current thread tile
}
#pragma unroll
for (int n = 0; n < TN; n += 4) {
    FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]); // Offset to current thread tile
}
```

Global memory cannot be directly written to shared memory and needs registers as intermediaries. The As writing process from global memory to register to shared memory is explicitly described, while the Bs writing process does not need register participation, but the compiler hides this code. The purpose of explicitly using registers for As caching is to transpose As. After transposition, a column before transposition becomes a row, which is memory-continuous and convenient for float4 reading.

![kernel_1](./images/kernel_5_vs_6.png)

Actual testing shows that overall calculation efficiency has increased.

## kernel 7

**Data prefetching**

Single caching refers to allocating a single shared memory block to cache global data, and allocating a single register memory block to cache shared data. Single caching cannot achieve parallel reading and storing because there is dependency between data. For example, in a single cache scenario, calculation depends on shared memory data. To ensure that global memory is completely stored in shared memory before calculation, a synchronization is required. Similarly, because calculation depends on shared memory data, another synchronization is required before storing new round of global memory to shared memory to ensure the previous round of calculation is completed.

Double buffering allocates double storage space, separating reading and writing. While calculating data reads from one storage space, it can simultaneously write the data needed for the next round to another memory. Therefore, only one synchronization is needed to ensure that the shared memory to be read is completed before calculation.

> Double buffering makes read and write synchronous, achieving data prefetching and hiding memory latency.

![](./images/describe_kernel_7.png)

![](./images/kernel_6_vs_7.png)

Using double buffering technology to achieve data prefetching, calculation efficiency has been further improved.

![](./images/kernel_culas_vs_7.png)

It basically approaches the calculation efficiency of CUBLAS official matrix multiplication.
