#pragma once

void MallocMatrix(int M, int N, int K, float*& A, float*& B, float*& C, float*& GT);

void FreeMatrix(float*& A, float*& B, float*& C, float*& GT);

void InitABCGT(int M, int N, int K, float* A, float* B, float* C, float* GT);

void PrintABC(int M, int N, int K, float* A, float* B, float* C);

void CheckResult(int M, int N, float* C, float* GT, float tolerance);