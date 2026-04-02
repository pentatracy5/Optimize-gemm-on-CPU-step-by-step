#pragma once

void ClearCache();

void MallocMatrix(const int M, const int N, const int K, int& lda, int& ldb, int& ldc, float*& A, float*& B, float*& C, float*& REF);

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF);

void InitABCREF(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C, float* REF);

void PrintABC(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C);

void CheckResult(const int M, const int N, const int ldc, float* C, float* REF, float tolerance);