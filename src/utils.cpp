#include <utils.h>
#include <algorithm>
#include <iostream>
#include <random>

using std::fill;
using std::cout;
using std::endl;
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;

void MallocMatrix(int M, int N, int K, float*& A, float*& B, float*& C, float*& REF)
{
	A = (float*)malloc(sizeof(float) * M * K);
	B = (float*)malloc(sizeof(float) * K * N);
	C = (float*)malloc(sizeof(float) * M * N);
	REF = (float*)malloc(sizeof(float) * M * N);
}

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF)
{
	free(A);
	free(B);
	free(C);
	free(REF);
	A = nullptr;
	B = nullptr;
	C = nullptr;
	REF = nullptr;
}

void InitABCREF(int M, int N, int K, float* A, float* B, float* C, float* REF)
{
	mt19937 engine(random_device{}());
	uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (size_t i = 0; i < M * K; i++)
		A[i] = dist(engine);
	for (size_t i = 0; i < K * N; i++)
		B[i] = dist(engine);
	fill(C, C + M * N, 0);
	fill(REF, REF + M * N, 0);
}

void PrintABC(int M, int N, int K, float* A, float* B, float* C)
{
	cout << "Matrix A:" << endl;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t k = 0; k < K; k++)
			cout << A[i * K + k] << " ";
		cout << endl;
	}

	cout << "Matrix B:" << endl;
	for (size_t k = 0; k < K; k++)
	{
		for (size_t j = 0; j < N; j++)
			cout << B[k * N + j] << " ";
		cout << endl;
	}

	cout << "Matrix C:" << endl;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
			cout << C[i * N + j] << " ";
		cout << endl;
	}
}

void CheckResult(int M, int N, float* C, float* REF, float tolerance)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++) 
			if (fabs(C[i * N + j] - REF[i * N + j]) > tolerance)
			{
				cout << "Error: C(" << i << ", " << j << ") = " << C[i * N + j] << ", but expected " << REF[i * N + j] << endl;
				return;
			}
	cout << "Check passed!" << endl;
	return;
}
