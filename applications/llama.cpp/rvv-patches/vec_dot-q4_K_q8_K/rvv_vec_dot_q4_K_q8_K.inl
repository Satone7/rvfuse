// rvv_vec_dot_q4_K_q8_K.inl — RVV VLEN=512 implementation of vec_dot_q4_K_q8_K
//
// Single source of truth for the RVV VLEN=512 vectorized kernel.
// Based on the VLEN=256 intrinsics implementation in arch/riscv/quants.c,
// extended for 64-byte vector registers (VLEN=512).
//
// Prerequisites before including this file:
//   - QK_K (256), K_SCALE_SIZE (12) must be defined
//   - block_q4_K, block_q8_K structures must be defined
//   - GGML_RESTRICT, GGML_UNUSED, GGML_CPU_FP16_TO_FP32 must be defined
//   - __riscv_v_intrinsic must be defined (compiling with -march=rv64gcv)
//   - <riscv_vector.h> must be included
//
// This file provides a code block for case 512 in the switch(vector_length)
// in ggml_vec_dot_q4_K_q8_K within arch/riscv/quants.c.
//
// Algorithm overview:
//   For each QK_K=256-element super-block:
//     1. Compute super-block scale d = x[i].d * y[i].d
//     2. Compute minimum correction dmin = x[i].dmin * y[i].d
//     3. Decode 6-bit packed scales and mins from 12 bytes
//     4. Process minimums: sumf -= dmin * sum(bsums[j] * mins[j/2])
//     5. For each of 4 subblocks (QK_K/64):
//        a. Load 32 bytes of Q4 data (4-bit packed, 2 elements/byte)
//        b. Extract lower nibbles → wmul with Q8[0..31] → reduce
//        c. Extract upper nibbles → wmul with Q8[32..63] → reduce
//        d. Apply per-subblock scale, accumulate
//     6. sumf += d * (sum_lo + sum_hi)
//
// VLEN=512 register mapping (E8, LMUL=1 → 64 elements per register):
//   - 32 bytes of Q4 fits in one register (half register width)
//   - 32 bytes of Q8 fits in one register (half register width)
//   - vwmul produces int16 (LMUL=2 for 32 elements)
//   - vredsum reduces to int16 (LMUL=1)

    case 512:
        for (int i = 0; i < nb; ++i) {

            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            // --- Decode 6-bit packed scales and mins ---
            memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            const uint8_t * restrict scales = (const uint8_t*)&utmp[0];
            const uint8_t * restrict mins   = (const uint8_t*)&utmp[2];

            // --- Process minimums (scalar, 8 elements — not worth vectorizing) ---
            {
                int sumi = 0;
                for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
                sumf -= dmin * sumi;
            }

            // --- Main dot product computation ---
            const uint8_t * restrict q4 = x[i].qs;
            const int8_t  * restrict q8 = y[i].qs;

            int32_t sum_1 = 0;
            int32_t sum_2 = 0;

            const size_t vl = 32; // 32 × int8 per vector register

            vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

            for (int j = 0; j < QK_K/64; ++j) {
                // Load 32 bytes of packed Q4
                vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

                // --- Lower nibbles × Q8[0..31] ---
                vint8m1_t q8_lo = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t q4_lo = __riscv_vreinterpret_v_u8m1_i8m1(
                    __riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
                vint16m2_t qv_lo = __riscv_vwmul_vv_i16m2(q4_lo, q8_lo, vl);
                vint16m1_t vs_lo = __riscv_vredsum_vs_i16m2_i16m1(qv_lo, vzero, vl);
                sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_lo) * scales[2*j + 0];

                // --- Upper nibbles × Q8[32..63] ---
                vint8m1_t q8_hi = __riscv_vle8_v_i8m1(q8 + 32, vl);
                vint8m1_t q4_hi = __riscv_vreinterpret_v_u8m1_i8m1(
                    __riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
                vint16m2_t qv_hi = __riscv_vwmul_vv_i16m2(q4_hi, q8_hi, vl);
                vint16m1_t vs_hi = __riscv_vredsum_vs_i16m2_i16m1(qv_hi, vzero, vl);
                sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_hi) * scales[2*j + 1];

                q4 += 32;
                q8 += 64;
            }

            sumf += d * (sum_1 + sum_2);
        }
        break;
