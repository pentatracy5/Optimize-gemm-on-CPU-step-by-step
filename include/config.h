#pragma once

#define NREPEATS 128
#define TOLERANCE (4e-2 * NREPEATS)
#define WARMUP 2

#define GEMM_Mblock 2048
#define GEMM_Nblock 3072
#define GEMM_Kblock 160

#define GEMM_NC 4096
#define GEMM_MC 1024
#define GEMM_KC 320
#define GEMM_NR 16
#define GEMM_MR 4

#define GEMM_NC_OMP 96
#define GEMM_MC_OMP 768
#define GEMM_KC_OMP 320
#define GEMM_NR_OMP 16
#define GEMM_MR_OMP 4

#define PROFILE

#define USE_OMP
#ifdef USE_OMP
#define OMP_THREADS 12
#else
#define OMP_THREADS 1
#endif