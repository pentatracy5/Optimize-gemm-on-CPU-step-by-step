#pragma once

void MallocMatrix(const size_t M, const size_t N, const size_t K, float*& A, float*& B, float*& C, float*& REF);

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF);

void InitABCREF(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C, float* REF);

void PrintABC(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C);

void CheckResult(const size_t M, const size_t N, float* C, float* REF, float tolerance);