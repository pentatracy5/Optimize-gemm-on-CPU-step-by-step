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

void MallocMatrix(int M, int N, int K, float*& A, float*& B, float*& C, float*& GT)
{
	A = (float*)malloc(sizeof(float) * M * K);
	B = (float*)malloc(sizeof(float) * K * N);
	C = (float*)malloc(sizeof(float) * M * N);
	GT = (float*)malloc(sizeof(float) * M * N);
}

void FreeMatrix(float*& A, float*& B, float*& C, float*& GT)
{
	free(A);
	free(B);
	free(C);
	free(GT);
	A = nullptr;
	B = nullptr;
	C = nullptr;
	GT = nullptr;
}

void InitAB(int M, int N, int K, float* A, float* B)
{
	mt19937 engine(random_device{}());
	uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (size_t i = 0; i < M * K; i++)
		A[i] = dist(engine);
	for (size_t i = 0; i < K * N; i++)
		B[i] = dist(engine);
}

void InitC(int M, int N, float* C)
{
	fill(C, C + M * N, 0);
}

void PrintABC(int M, int N, int K, float* A, float* B, float* C)
{
	cout << "Matrix A:" << endl;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t k = 0; k < K; k++)
			cout << A(i, k) << " ";
		cout << endl;
	}

	cout << "Matrix B:" << endl;
	for (size_t k = 0; k < K; k++)
	{
		for (size_t j = 0; j < N; j++)
			cout << B(k, j) << " ";
		cout << endl;
	}

	cout << "Matrix C:" << endl;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
			cout << C(i, j) << " ";
		cout << endl;
	}
}

void CheckResult(int M, int N, float* C, float* GT)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++) 
			if (fabs(C(i, j) - GT(i, j)) > TOLERANCE)
			{
				cout << "Error: C(" << i << ", " << j << ") = " << C(i, j) << ", but expected " << GT[i * N + j] << endl;
				return;
			}
	cout << "Check passed!" << endl;
	return;
}
