#pragma once

#define NREPEATS 128
#define TOLERANCE (1e-1 * NREPEATS)
#define WARMUP 2

#define GEMM_Mblock 2048
#define GEMM_Nblock 3072
#define GEMM_Kblock 160

#define USE_OMP

#ifdef USE_OMP

#define OMP_THREADS 12

//#define GEMM_NC 96
//#define GEMM_MC 768
//#define GEMM_KC 320
//#define GEMM_NR 16
//#define GEMM_MR 4

//#define GEMM_NC 192
//#define GEMM_MC 768
//#define GEMM_KC 160
//#define GEMM_NR 24
//#define GEMM_MR 4

#define GEMM_NC 4080
#define GEMM_MC 72
#define GEMM_KC 256
#define GEMM_NR 24
#define GEMM_MR 4

#else

#define OMP_THREADS 1

#define GEMM_NC 4096
#define GEMM_MC 1024
#define GEMM_KC 320
#define GEMM_NR 16
#define GEMM_MR 4

#endif

#define PROFILE