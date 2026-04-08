#pragma once

constexpr bool PROFILE = true;
constexpr bool PROFILEREF = false;
constexpr int WARMUP = 2;
constexpr int NREPEATS = PROFILE ? 128 : 2;
constexpr float TOLERANCE = 0.08f * NREPEATS;

#define USE_OMP
#ifdef USE_OMP

constexpr int OMP_THREADS = 16;

// for MatMul09 - MatMul10 (v0) (blis version)
//constexpr int GEMM_NR = 24;
//constexpr int GEMM_MR = 4;
//constexpr int GEMM_KC = 256;
//constexpr int GEMM_MC = 32 * GEMM_MR; // 128
//constexpr int GEMM_NC = 11 * GEMM_NR; // 264

// for MatMul09 - MatMul10 (v1)
//constexpr int GEMM_NR = 24;
//constexpr int GEMM_MR = 4;
//constexpr int GEMM_KC = 240;
//constexpr int GEMM_MC = 48 * GEMM_MR; // 192
//constexpr int GEMM_NC = 40 * GEMM_NR; // 960

// for MatMul09 - MatMul10 (v2) (blis version)
//constexpr int GEMM_NR = 16;
//constexpr int GEMM_MR = 6;
//constexpr int GEMM_KC = 256;
//constexpr int GEMM_MC = 24 * GEMM_MR; // 144
//constexpr int GEMM_NC = 255 * GEMM_NR; // 4080

// for MatMul09 - MatMul10 (v3)
constexpr int GEMM_NR = 16;
constexpr int GEMM_MR = 6;
constexpr int GEMM_KC = 256;
constexpr int GEMM_MC = 8 * GEMM_MR; // 48
constexpr int GEMM_NC = 256 * GEMM_NR; // 4096

#else

constexpr int OMP_THREADS = 1;

//for MatMul01 - MatMul05
//constexpr int GEMM_NC = 5120;
//constexpr int GEMM_KC = 80;
//constexpr int GEMM_MC = 1280;

//for MatMul06 - MatMul07
//constexpr int GEMM_NR = 24;
//constexpr int GEMM_MR = 4;
//constexpr int GEMM_KC = 320;
//constexpr int GEMM_MC = 320 * GEMM_MR; // 1280
//constexpr int GEMM_NC = 128 * GEMM_NR; // 3072

//for MatMul08
constexpr int GEMM_NR = 24;
constexpr int GEMM_MR = 4;
constexpr int GEMM_KC = 320;
constexpr int GEMM_MC = 320 * GEMM_MR; // 1280
constexpr int GEMM_NC = 64 * GEMM_NR; // 1536

#endif