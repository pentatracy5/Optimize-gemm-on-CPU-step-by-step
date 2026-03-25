#pragma once

typedef void (*MatMulFunc)(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMulREF(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul0(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul1(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul2(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul3(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul4(size_t M, size_t N, size_t K, float* A, float* B, float* C); 

void MatMul5(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul6(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul7(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void MatMul8(size_t M, size_t N, size_t K, float* A, float* B, float* C);

static const MatMulFunc matmulFuncs[] = { MatMul0, MatMul1, MatMul2, MatMul3, MatMul4, MatMul5, MatMul6, MatMul7, MatMul8 };