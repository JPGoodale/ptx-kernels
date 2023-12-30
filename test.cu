__global__ void matrixMulAsm(int* A, int* B, int* C, int N) {
    asm volatile(
            ".reg .b64 %%td0, %%td1, %%td2, %%td3, %%td4, %%td5, %%td6, %%td7, %%td8; \n\t"
            ".reg .b32 %%t0, %%t1, %%t2, %%t3, %%t4, %%t5, %%t6, %%t7, %%t8, %%t9, %%t10, %%t11, %%t12, %%t13, %%t14, %%t15, %%t16, %%t17, %%t18, %%t19, %%t20, %%t21; \n\t"
            ".reg .pred %%pt0, %%pt1, %%pt2, %%pt3; \n\t"

            // convert input params to global addresses
            "cvta.to.global.u64 %%td0, %0; \n\t"
            "cvta.to.global.u64 %%td1, %1; \n\t"
            "cvta.to.global.u64 %%td2, %2; \n\t"

            // calculate row idx
            "mov.u32 %%t0, %%tid.y; \n\t"
            "mov.u32 %%t1, %%ctaid.y; \n\t"
            "mov.u32 %%t2, %%ntid.y; \n\t"
            "mad.lo.s32 %%t3, %%t2, %%t1, %%t0; \n\t"

            // calculate column idx
            "mov.u32 %%t4, %%tid.x; \n\t"
            "mov.u32 %%t5, %%ctaid.x; \n\t"
            "mov.u32 %%t6, %%ntid.x; \n\t"
            "mad.lo.s32 %%t7, %%t6, %%t5, %%t4; \n\t"

            // break loop if N >= column idx OR row idx
            "setp.ge.s32 %%pt0, %%t3, %3; \n\t"
            "setp.ge.s32 %%pt1, %%t7, %3; \n\t"
            "or.pred %%pt2, %%pt1, %%pt0; \n\t"
            "@%%pt2 bra $Break; \n\t"

            // N - 1
            "add.s32 %%t8, %3, -1; \n\t"
            // N % 4
            "and.b32 %%t9, %3, 3; \n\t"
            // N - (N % 4)
            "sub.s32 %%t10, %3, %%t9; \n\t"
            // N * 4
            "mul.wide.s32 %%td3, %3, 4; \n\t"

            // A_idx = row_idx * N * 4 bytes
            "mul.lo.s32 %%t11, %%t3, %3; \n\t"
            "mul.wide.s32 %%td4, %%t11, 4; \n\t"
            // offset starting address of A by new idx
            "add.s64 %%td5, %%td4, %%td0; \n\t"
            // offset A_address further by 8 bytes
            "add.s64 %%td6, %%td5, 8; \n\t"

            // k = 0
            "mov.u32 %%t12, 0; \n\t"
            "mov.u32 %%t13, %%t12; \n\t"

            // Loop
            "$Loop: \n\t"
            // load A_address - 8 bytes into register
            "\t ld.global.u32 %%t14, [%%td6+-8]; \n\t"

            // A register + k
            "\t add.s32 %%t15, %%t14, %%t13; \n\t"

            // load A_address - 4 bytes into register
            "\t ld.global.u32 %%t16, [%%td6+-4]; \n\t"

            // (A_address - 4 bytes) + ((A_address - 8 bytes) + k)
            "\t add.s32 %%t17, %%t16, %%t15; \n\t"

            // load A_address into register
            "\t ld.global.u32 %%t18, [%%td6]; \n\t"

            // A_address + ((A_address - 4 bytes) + ((A_address - 8 bytes) + k))
            "\t add.s32 %%t19, %%t18, %%t17; \n\t"

            // load A_address + 4 bytes into register
            "\t ld.global.u32 %%t20, [%%td6+4]; \n\t"

            // A = A_address + 4 bytes + (A_address + ((A_address - 4 bytes) + ((A address - 8 bytes) + k)))
            "\t add.s32 %%t13, %%t20, %%t19; \n\t"

            // 0 + 4
            "\t add.s32 %%t12, %%t12, 4; \n\t"

            // A_address = A_address + 16
            "\t add.s64 %%rd6, %%rd6, 16; \n\t"

            // (N % 4) - 4
            "\t add.s32 %%t10, %%t10, -4; \n\t"

            // if ((N % 4) - 4) != 0: rerun loop
            "\t setp.ne.s32 %%pt3, %%t10, 0; \n\t"
            "\t @%%pt3 bra $Loop; \n\t"

            // C_idx = ((N * row_idx) + column_idx) * 4 bytes
            "add.s32 %%t21, %%t11, %%t7; \n\t"
            "mul.wide.s32 %%td7, %%t21, 4; \n\t"
            // C_address = C_address + C_idx
            "add.s64 %%td8, %%td7, %%td2; \n\t"
            // store results in C_address
            "st.global.u32 [%%td8], %%t13; \n\t"

            "$Break: ret; \n\t"
            :: "l" (A), "l" (B), "l" (C), "r" (N)
            : "memory"
            );
}
