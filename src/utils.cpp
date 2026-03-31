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

void MallocMatrix(const size_t M, const size_t N, const size_t K, float*& A, float*& B, float*& C, float*& REF)
{
	A = (float*)_aligned_malloc(sizeof(float) * M * K, 32);
	B = (float*)_aligned_malloc(sizeof(float) * K * N, 32);
	C = (float*)_aligned_malloc(sizeof(float) * M * N, 32);
	REF = (float*)_aligned_malloc(sizeof(float) * M * N, 32);
}

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF)
{
	_aligned_free(A);
	_aligned_free(B);
	_aligned_free(C);
	_aligned_free(REF);
	A = nullptr;
	B = nullptr;
	C = nullptr;
	REF = nullptr;
}

void InitABCREF(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C, float* REF)
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

void PrintABC(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
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

void CheckResult(const size_t M, const size_t N, float* C, float* REF, float tolerance)
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
