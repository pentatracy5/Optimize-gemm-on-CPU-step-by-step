#pragma once



#define PROFILE

#define WARMUP 2
#define TOLERANCE (8e-2 * NREPEATS)

#ifdef PROFILE

#define NREPEATS 128

#else

#define NREPEATS 2

#endif



#define USE_OMP

#ifdef USE_OMP

#define OMP_THREADS 16

//for MatMul09 (v0) (blis version)
//#define GEMM_NR 24
//#define GEMM_MR 4
//#define GEMM_KC 256
//#define GEMM_MC (32 * GEMM_MR) // 128
//#define GEMM_NC (11 * GEMM_NR) // 264

//for MatMul09 (v1)
//#define GEMM_NR 24
//#define GEMM_MR 4
//#define GEMM_KC 240
//#define GEMM_MC (48 * GEMM_MR) // 192
//#define GEMM_NC (40 * GEMM_NR) // 960

//for MatMul09 (v2) (blis version)
//#define GEMM_NR 16
//#define GEMM_MR 6
//#define GEMM_KC 256
//#define GEMM_MC (24 * GEMM_MR) // 144
//#define GEMM_NC (255 * GEMM_NR) // 4080

//for MatMul09 (v3)
#define GEMM_NR 16
#define GEMM_MR 6
#define GEMM_KC 256
#define GEMM_MC (8 * GEMM_MR) // 48
#define GEMM_NC (256 * GEMM_NR) // 4096

#else

#define OMP_THREADS 1

//for MatMul01 - MatMul05
//#define GEMM_NC 5120
//#define GEMM_KC 80
//#define GEMM_MC 1280

//for MatMul06 - MatMul07
//#define GEMM_NR 24
//#define GEMM_MR 4
//#define GEMM_KC 320
//#define GEMM_MC (320 * GEMM_MR) // 1280
//#define GEMM_NC (128 * GEMM_NR) // 3072

//for MatMul08
#define GEMM_NR 24
#define GEMM_MR 4
#define GEMM_KC 320
#define GEMM_MC (320 * GEMM_MR) // 1280
#define GEMM_NC (64 * GEMM_NR) // 1536

#endif