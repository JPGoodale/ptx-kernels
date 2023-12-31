#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>

// Regular C++ kernel
__global__ void matrixMul(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            temp_sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = temp_sum;
    }
}


__global__ void matrixMulAsm(int* A, int* B, int* C, int N) {
    asm volatile(
            ".reg .b64 %%ad0, %%ad1, %%ad2, %%ad3, %%ad4, %%ad5, %%ad6, %%ad7, %%ad8, %%ad9, %%ad10, %%ad11, %%ad12, %%ad13, %%ad14, %%ad15; \n\t"
            ".reg .b32 %%t0, %%t1, %%t2, %%t3, %%t4, %%t5, %%t6, %%t7, %%t8, %%t9, %%t10, %%t11, %%t12, %%t13, %%t14, %%t15, %%t16, %%t17, %%t18, %%t19, %%t20, %%t21, %%t22, %%t23; \n\t"
            ".reg .pred %%pr0, %%pr1, %%pr2, %%pr3; \n\t"

            // convert input params to global addresses
            "cvta.to.global.u64 %%ad0, %0; \n\t"
            "cvta.to.global.u64 %%ad1, %1; \n\t"
            "cvta.to.global.u64 %%ad2, %2; \n\t"

            // calculate row_idx
            "mov.u32 %%t0, %%tid.y; \n\t"
            "mov.u32 %%t1, %%ctaid.y; \n\t"
            "mov.u32 %%t2, %%ntid.y; \n\t"
            "mad.lo.s32 %%t3, %%t2, %%t1, %%t0; \n\t"

            // calculate column_idx
            "mov.u32 %%t4, %%tid.x; \n\t"
            "mov.u32 %%t5, %%ctaid.x; \n\t"
            "mov.u32 %%t6, %%ntid.x; \n\t"
            "mad.lo.s32 %%t7, %%t6, %%t5, %%t4; \n\t"

            // break loop if N >= column or row indices
            "setp.ge.s32 %%pr0, %%t3, %3; \n\t"
            "setp.ge.s32 %%pr1, %%t7, %3; \n\t"
            "or.pred %%pr2, %%pr1, %%pr0; \n\t"
            "@%%pr2 bra $Break; \n\t"

            // compute row_idx * N (will be used again for C)
            "mul.lo.s32 %%t8, %%t3, %3; \n\t"
            // A_idx = row_idx * N * 4 bytes
            "mul.wide.s32 %%ad4, %%t8, 4; \n\t"
            // offset starting address of A by new idx
            "add.s64 %%ad5, %%ad4, %%ad0; \n\t"
            // offset A_addr further by 8 bytes
            "add.s64 %%ad6, %%ad5, 8; \n\t"

            // B_idx = column_idx * 4 bytes
            "mul.wide.s32 %%ad7, %%t7, 4; \n\t"
            // offset B_addr by new idx
            "add.s64 %%ad8, %%ad7, %%ad1; \n\t"

            // move N into a register
            "mov.s32 %%t9, %3; \n\t"
            // N * 4 (N elements of 4 bytes each, this is used for moving the
            // B_addr to the corasponding row element in the next column)
            "mul.wide.s32 %%ad9, %3, 4; \n\t"
            // set temp_sum = 0
            "mov.u32 %%t10, 0; \n\t"

            // some loop unrolling is used here: https://en.wikipedia.org/wiki/Loop_unrolling
            // by loading and processing 4 elements at a time
            "$Loop: \n\t"
            // load A_addr - 8 btyes into register
            "ld.global.u32 %%t11, [%%ad6+-8]; \n\t"
            // load B_addr into register
            "ld.global.u32 %%t12, [%%ad8]; \n\t"
            // temp_sum = A * B + temp_sum
            "mad.lo.s32 %%t13, %%t12, %%t11, %%t10; \n\t"
            // new B_addr = B_addr + (N*4)
            "add.s64 %%ad10, %%ad8, %%ad9; \n\t"

            // load A_addr - 4 bytes
            "ld.global.u32 %%t14, [%%ad6+-4]; \n\t"
            // load B_addr
            "ld.global.u32 %%t15, [%%ad10]; \n\t"
            // temp_sum = A * B + temp_sum
            "mad.lo.s32 %%t16, %%t15, %%t14, %%t13; \n\t"
            // new B_addr = B_addr + (N*4)
            "add.s64 %%ad11, %%ad10, %%ad9; \n\t"

            // load A_addr
            "ld.global.u32 %%t17, [%%ad6]; \n\t"
            // load B_addr
            "ld.global.u32 %%t18, [%%ad11]; \n\t"
            // temp_sum = A * B + temp_sum
            "mad.lo.s32 %%t19, %%t18, %%t17, %%t16; \n\t"
            // new B_addr = B_addr + (N*4)
            "add.s64 %%ad12, %%ad11, %%ad9; \n\t"

            // set original B_addr = B_addr + (N*4)
            "add.s64 %%ad8, %%ad12, %%ad9; \n\t"

            // load A_addr + 4 bytes
            "ld.global.u32 %%t20, [%%ad6+4]; \n\t"
            // Load B_addr
            "ld.global.u32 %%t21, [%%ad12]; \n\t"
            // temp_sum (now back to the register we started with) = A * B + temp_sum
            "mad.lo.s32 %%t10, %%t21, %%t20, %%t19; \n\t"
            // new A_addr = A_addr + 16
            "add.s64 %%ad6, %%ad6, 16; \n\t"

            // N = N - 4
            "add.s32 %%t9, %%t9, -4; \n\t"
            // if N != 0: rerun loop
            "setp.ne.s32 %%pr3, %%t9, 0; \n\t"
            "@%%pr3 bra $Loop;"

            // (C_idx = row_idx * N + column_idx) * 4 bytes
            "add.s32 %%t23, %%t8, %%t7; \n\t"
            "mul.wide.s32 %%ad13, %%t23, 4; \n\t"
            // offset C_addr by C_idx
            "add.s64 %%ad14, %%ad13, %%ad2; \n\t"
            // store results into C_addr
            "st.global.u32 [%%ad14], %%t10; \n\t"

            "$Break: ret; \n\t"
            :: "l" (A), "l" (B), "l" (C), "r" (N)
            : "memory"
            );
}


void matrix_init(int* A, int* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = rand() % 100;
            B[i * N + j] = rand() % 100;
        }
    }
}


void verify_result(int* A, int* B, int* C, int N) {
    int *verify_c;
    verify_c = (int*)malloc(N*N*sizeof(int));
    memset(verify_c, 0, N*N*sizeof(int));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                verify_c[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(C[i * N + j] == verify_c[i * N + j]);
        }
    }
}


void matmul() {
    int N = 1 << 10;
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t bytes = N*N*sizeof(int);

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    matrix_init(h_a, h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int)ceil(N / BLOCK_SIZE);

    dim3 blocks(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    matrixMulAsm<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    printf("%i \n", *h_c);

    verify_result(h_a, h_b, h_c, N);
    printf("COMPLETED SUCCESSFULLY\n");
}
