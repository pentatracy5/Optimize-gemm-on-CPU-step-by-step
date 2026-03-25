#pragma once

void MallocMatrix(size_t M, size_t N, size_t K, float*& A, float*& B, float*& C, float*& REF);

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF);

void InitABCREF(size_t M, size_t N, size_t K, float* A, float* B, float* C, float* REF);

void PrintABC(size_t M, size_t N, size_t K, float* A, float* B, float* C);

void CheckResult(size_t M, size_t N, float* C, float* REF, float tolerance);