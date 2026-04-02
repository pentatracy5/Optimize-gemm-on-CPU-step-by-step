#pragma once

typedef void (*MatMulFunc)(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMulREF(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul00(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul01(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul02(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul03(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul04(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul05(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul06(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul07(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul08(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul09(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void MatMul10(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

static const MatMulFunc matmulFuncs[] = { MatMul00, MatMul01, MatMul02, MatMul03, MatMul04, MatMul05, MatMul06, MatMul07, MatMul08, MatMul09, MatMul10 };