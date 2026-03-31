#pragma once

typedef void (*MatMulFunc)(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMulREF(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul0(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul1(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul2(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul3(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul4(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C); 

void MatMul5(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul6(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul7(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul8(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void MatMul9(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

static const MatMulFunc matmulFuncs[] = { MatMul0, MatMul1, MatMul2, MatMul3, MatMul4, MatMul5, MatMul6, MatMul7, MatMul8, MatMul9 };