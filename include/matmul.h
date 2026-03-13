#pragma once

typedef void (*MatMulFunc)(int M, int N, int K, float* A, float* B, float* C);

void MatMulGT(int M, int N, int K, float* A, float* B, float* C);

void MatMul0(int M, int N, int K, float* A, float* B, float* C);

void MatMul1(int M, int N, int K, float* A, float* B, float* C);