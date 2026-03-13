#pragma once

void MallocMatrix(int M, int N, int K, float*& A, float*& B, float*& C, float*& REF);

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF);

void InitABCREF(int M, int N, int K, float* A, float* B, float* C, float* REF);

void PrintABC(int M, int N, int K, float* A, float* B, float* C);

void CheckResult(int M, int N, float* C, float* REF, float tolerance);