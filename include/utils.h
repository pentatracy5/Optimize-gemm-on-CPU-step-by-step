#pragma once

void ClearCache();

void MallocMatrix(const size_t M, const size_t N, const size_t K, size_t& lda, size_t& ldb, size_t& ldc, float*& A, float*& B, float*& C, float*& REF);

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF);

void InitABCREF(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C, float* REF);

void PrintABC(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C);

void CheckResult(const size_t M, const size_t N, const size_t ldc, float* C, float* REF, float tolerance);