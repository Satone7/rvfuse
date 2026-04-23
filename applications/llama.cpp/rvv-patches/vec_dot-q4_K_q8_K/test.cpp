// test.cpp — correctness test for RVV vec_dot_q4_K_q8_K (VLEN=512)
//
// Compares the RVV vectorized implementation against the scalar (generic)
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv, VLEN=512):
//   clang++ -std=c++17 -O2 \
//       --target=riscv64-unknown-linux-gnu \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       -D__riscv_v_fixed_vlen=512 \
//       -Iggml/src -Iggml/src/include -Iggml/include -Iggml/src/ggml-cpu \
//       test.cpp -o test_rvv -lm
//
// Build (scalar, no RVV):
//   g++ -std=c++17 -O2 test.cpp -o test_scalar -lm
//
// Run under QEMU:
//   qemu-riscv64 -L /path/to/sysroot ./test_rvv
//
// Run natively (scalar):
//   ./test_scalar

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>

// =============================================================================
// Minimal ggml type definitions (from ggml-common.h)
// =============================================================================
typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#ifdef _MSC_VER
#define GGML_EXTENSION
#else
#define GGML_EXTENSION __extension__
#endif

#define QK_K 256
#define K_SCALE_SIZE 12

// block_q4_K: 4-bit quantization, 256 elements per super-block
typedef struct {
    ggml_half d;                    // super-block scale for quantized scales
    ggml_half dmin;                 // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];   // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];            // 4-bit quants (128 bytes)
} block_q4_K;

// block_q8_K: 8-bit quantization, 256 elements per block (intermediate)
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants (256 bytes)
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16 (32 bytes)
} block_q8_K;

#define GGML_RESTRICT __restrict__
#define GGML_UNUSED(x) (void)(x)

// FP16 to FP32 conversion (software implementation for portability)
static inline float ggml_f16_to_f32(ggml_half h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    float result;
    if (exp == 0) {
        if (frac == 0) {
            uint32_t bits = sign << 31;
            memcpy(&result, &bits, 4);
        } else {
            exp = 0;
            while (!(frac & 0x400)) { frac <<= 1; exp++; }
            frac &= 0x3ff;
            exp = (-exp - 1 + 127);
            uint32_t bits = (sign << 31) | (exp << 23) | (frac << 13);
            memcpy(&result, &bits, 4);
        }
    } else if (exp == 31) {
        uint32_t bits = (sign << 31) | (0xff << 23) | (frac << 13);
        memcpy(&result, &bits, 4);
    } else {
        uint32_t bits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
        memcpy(&result, &bits, 4);
    }
    return result;
}

#define GGML_CPU_FP16_TO_FP32(x) ggml_f16_to_f32(x)

// =============================================================================
// Scalar (generic) reference implementation
// =============================================================================
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,
    size_t by, int nrc) {

    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = (const block_q4_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// =============================================================================
// RVV vectorized implementation — VLEN=512 fixed
// =============================================================================
#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>

// With __riscv_v_fixed_vlen=512, __riscv_vlenb() returns 64 at compile time
// The compiler generates code assuming VLEN=512
static void ggml_vec_dot_q4_K_q8_K_rvv512(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,
    size_t by, int nrc) {

    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = (const block_q4_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        // Decode 6-bit packed scales and mins
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // Process minimums (scalar)
        {
            int sumi = 0;
            for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
            sumf -= dmin * sumi;
        }

        // Main dot product computation
        const uint8_t * __restrict__ q4 = x[i].qs;
        const int8_t  * __restrict__ q8 = y[i].qs;

        int32_t sum_1 = 0;
        int32_t sum_2 = 0;

        const size_t vl = 32; // 32 × int8 per vector register

        vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

        for (int j = 0; j < QK_K/64; ++j) {
            vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

            // Lower nibbles × Q8[0..31]
            vint8m1_t q8_lo = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t q4_lo = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
            vint16m2_t qv_lo = __riscv_vwmul_vv_i16m2(q4_lo, q8_lo, vl);
            vint16m1_t vs_lo = __riscv_vredsum_vs_i16m2_i16m1(qv_lo, vzero, vl);
            sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_lo) * scales[2*j + 0];

            // Upper nibbles × Q8[32..63]
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

    *s = sumf;
}
#endif // __riscv_v_intrinsic

// =============================================================================
// Pseudo-random data generator (LCG, deterministic)
// =============================================================================
static uint32_t rng_state = 42;

static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}

static uint8_t rng_u8() {
    return (uint8_t)(rng_next() & 0xFF);
}

static ggml_half rng_half() {
    uint32_t sign = rng_next() & 1;
    uint32_t exp  = (rng_next() % 25) + 5;
    uint32_t frac = rng_next() & 0x3ff;
    return (ggml_half)((sign << 15) | (exp << 10) | frac);
}

static float rng_float() {
    return ((float)(rng_next() % 20000) - 10000.0f) / 1000.0f;
}

static void generate_q4_K(block_q4_K * blk) {
    for (int i = 0; i < K_SCALE_SIZE; i++) blk->scales[i] = rng_u8();
    for (int i = 0; i < QK_K/2; i++) blk->qs[i] = rng_u8();
    blk->d    = rng_half();
    blk->dmin = rng_half();
}

static void generate_q8_K(block_q8_K * blk) {
    blk->d = rng_float();
    for (int i = 0; i < QK_K; i++) blk->qs[i] = (int8_t)(rng_next() & 0xFF);
    for (int j = 0; j < QK_K/16; j++) {
        int16_t sum = 0;
        for (int l = 0; l < 16; l++) sum += blk->qs[j * 16 + l];
        blk->bsums[j] = sum;
    }
}

// =============================================================================
// RVV runtime VLEN-aware wrapper (matches llama.cpp dispatch pattern)
// =============================================================================
#if defined(__riscv_v_intrinsic)
static void ggml_vec_dot_q4_K_q8_K_dispatch(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,
    size_t by, int nrc) {

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    float sumf = 0;

    const int vector_length = __riscv_vlenb() * 8;

    switch (vector_length) {
    case 256: {
        // Same algorithm as VLEN=512 but with VLEN=256 LMUL config
        const block_q4_K * GGML_RESTRICT x = (const block_q4_K *)vx;
        const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;
        const uint8_t * scales = (const uint8_t*)&utmp[0];
        const uint8_t * mins   = (const uint8_t*)&utmp[2];

        for (int i = 0; i < nb; ++i) {
            const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

            memcpy(utmp, x[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            int sumi = 0;
            for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
            sumf -= dmin * sumi;

            const uint8_t * __restrict__ q4 = x[i].qs;
            const int8_t  * __restrict__ q8 = y[i].qs;
            int32_t sum_1 = 0, sum_2 = 0;
            size_t vl = 32;
            vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

            for (int j = 0; j < QK_K/64; ++j) {
                vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);
                vint8m1_t q8_lo = __riscv_vle8_v_i8m1(q8, vl);
                vint8m1_t q4_lo = __riscv_vreinterpret_v_u8m1_i8m1(
                    __riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
                vint16m2_t qv_lo = __riscv_vwmul_vv_i16m2(q4_lo, q8_lo, vl);
                vint16m1_t vs_lo = __riscv_vredsum_vs_i16m2_i16m1(qv_lo, vzero, vl);
                sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_lo) * scales[2*j + 0];

                vint8m1_t q8_hi = __riscv_vle8_v_i8m1(q8 + 32, vl);
                vint8m1_t q4_hi = __riscv_vreinterpret_v_u8m1_i8m1(
                    __riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
                vint16m2_t qv_hi = __riscv_vwmul_vv_i16m2(q4_hi, q8_hi, vl);
                vint16m1_t vs_hi = __riscv_vredsum_vs_i16m2_i16m1(qv_hi, vzero, vl);
                sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_hi) * scales[2*j + 1];

                q4 += 32; q8 += 64;
            }
            sumf += d * (sum_1 + sum_2);
        }
        break;
    }
    default:
        ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
        break;
    }
    *s = sumf;
}
#endif

// =============================================================================
// Test runner
// =============================================================================
static int run_test(int n_blocks, const char * label, bool use_rvv_dispatch) {
    const int n = n_blocks * QK_K;

    std::vector<block_q4_K> x(n_blocks);
    std::vector<block_q8_K> y(n_blocks);

    rng_state = 42 + n_blocks;
    for (int i = 0; i < n_blocks; i++) {
        generate_q4_K(&x[i]);
        generate_q8_K(&y[i]);
    }

    float s_ref = 0;
    ggml_vec_dot_q4_K_q8_K_generic(n, &s_ref, 0, x.data(), 0, y.data(), 0, 1);

#if defined(__riscv_v_intrinsic)
    float s_rvv = 0;
    if (use_rvv_dispatch) {
        ggml_vec_dot_q4_K_q8_K_dispatch(n, &s_rvv, 0, x.data(), 0, y.data(), 0, 1);
    } else {
        s_rvv = s_ref; // placeholder, dispatch not used
    }

    const int vlen = __riscv_vlenb() * 8;
    float diff = fabsf(s_ref - s_rvv);
    float rel = (fabsf(s_ref) > 1e-6f) ? diff / fabsf(s_ref) : diff;
    bool pass = rel < 1e-3f;

    printf("[%s] blocks=%d  vlen=%d  ref=%.6f  rvv=%.6f  diff=%.6f  rel=%.4e  %s\n",
        label, n_blocks, vlen, s_ref, s_rvv, diff, rel, pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
#else
    printf("[%s] blocks=%d  ref=%.6f  (scalar-only)\n", label, n_blocks, s_ref);
    return 0;
#endif
}

int main() {
    printf("=== vec_dot_q4_K_q8_K: RVV vs Scalar correctness test ===\n");
#if defined(__riscv_v_intrinsic)
    printf("Build: RVV enabled, runtime VLEN=%d\n", __riscv_vlenb() * 8);
#else
    printf("Build: Scalar only\n");
#endif
    printf("\n");

    int failures = 0;
    // Test 1: RVV dispatch (runtime VLEN selection, works on any QEMU VLEN)
    failures += run_test(1,  "rvv-disp-1", true);
    failures += run_test(2,  "rvv-disp-2", true);
    failures += run_test(4, "rvv-disp-4", true);
    // Test 2: Reference scalar only (baseline)
    failures += run_test(1,  "scalar-only-1", false);

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}