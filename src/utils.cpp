#include <utils.h>
#include <algorithm>
#include <iostream>

using std::fill;
using std::cout;
using std::endl;

void InitABC(int M, int N, int K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M * K; i++)
		A[i] = (float)(i % 4 + 1);
	for (size_t i = 0; i < K * N; i++)
		B[i] = (float)(i % 4 + 1);
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
			if (C(i, j) != GT(i, j))
			{
				cout << "Error: C(" << i << ", " << j << ") = " << C(i, j) << ", but expected " << GT[i * N + j] << endl;
				return;
			}
}
