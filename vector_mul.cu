#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>


// Regular CUDA version of kernel for reference
__global__ void vectorMul(int* a, int* b, int* c, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < N) {
        c[thread_id] = a[thread_id] * b[thread_id];
    }
}

__global__ void vectorMulAsm(int* a, int* b, int* c, int N) {
    asm volatile(
            ".reg .b64 %%td0, %%td1, %%td2, %%td3; \n\t"
            ".reg .b32 %%t0, %%t1, %%t2, %%t3, %%t4, %%t5, %%t6; \n\t"
            ".reg .pred %%pt0; \n\t"

            "cvta.to.global.u64 %%td0, %0; \n\t"
            "cvta.to.global.u64 %%td1, %1; \n\t"
            "cvta.to.global.u64 %%td2, %2; \n\t"

            "mov.u32 %%t0, %%tid.x; \n\t"
            "mov.u32 %%t1, %%ctaid.x; \n\t"
            "mov.u32 %%t2, %%ntid.x; \n\t"
            "mad.lo.s32 %%t3, %%t2, %%t1, %%t0; \n\t"

            "setp.ge.s32 %%pt0, %%t3, %3; \n\t"
            "@%%pt0 bra $Break; \n\t"

            "mul.wide.s32 %%td3, %%t3, 4; \n\t"
            "add.s64 %%td0, %%td3, %%td0; \n\t"
            "add.s64 %%td1, %%td3, %%td1; \n\t"
            "add.s64 %%td2, %%td3, %%td2; \n\t"

            "ld.global.s32 %%t4, [%%td0]; \n\t"
            "ld.global.s32 %%t5, [%%td1]; \n\t"
            "mul.lo.s32 %%t6, %%t4, %%t5; \n\t"
            "st.global.s32 [%%td2], %%t6; \n\t"

            "$Break: ret; \n\t"
            :: "l" (a), "l" (b), "l" (c), "r" (N)
            : "memory"
            );
}


void vector_init(int* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
    }
}


void error_check(int* a, int* b, int* c, int N) {
    for (int i = 0; i < N; i++) {
        assert(c[i] == a[i] * b[i]);
    }
}


void vector_mul() {
    int id = cudaGetDevice(&id);
    int N = 1 << 16;
    size_t bytes = sizeof(int) * N;

    int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    vector_init(a, N);
    vector_init(b, N);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (int)ceil(N / NUM_THREADS);

    cudaMemPrefetchAsync(a, bytes, id);
    cudaMemPrefetchAsync(b, bytes, id);

    vectorMulAsm <<<NUM_BLOCKS, NUM_THREADS>>> (a, b, c, N);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    error_check(a, b, c, N);
    printf("COMPLETED SUCCESSFULLY\n");

    cudaFree(a);
    cudaFree(b);
}
