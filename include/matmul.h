#pragma once

typedef void (*MatMulFunc)(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMulREF(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul00(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul01(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul02(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul03(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul04(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul05(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul06(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul07(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul08(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul09(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void MatMul10(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

static const MatMulFunc matmulFuncs[] = { MatMul00, MatMul01, MatMul02, MatMul03, MatMul04, MatMul05, MatMul06, MatMul07, MatMul08, MatMul09, MatMul10 };