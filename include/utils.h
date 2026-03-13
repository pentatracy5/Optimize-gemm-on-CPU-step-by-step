#pragma once

#define A(i, k) A[(i) * K + (k)] // M * K
#define B(k, j) B[(k) * N + (j)] // K * N
#define C(i, j) C[(i) * N + (j)] // M * N
#define GT(i, j) GT[(i) * N + (j)] // M * N, GT means ground truth

void InitABC(int M, int N, int K, float* A, float* B, float* C);

void PrintABC(int M, int N, int K, float* A, float* B, float* C);

void CheckResult(int M, int N, float* C, float* GT);