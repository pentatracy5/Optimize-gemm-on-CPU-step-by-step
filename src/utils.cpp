#include <config.h>
#include <utils.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

using std::fill;
using std::cout;
using std::endl;
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;
using std::vector;

vector<int> v;
void ClearCache()
{
	constexpr int cache_size = 32 * 1024 * 1024;
	constexpr int num_elements = cache_size / sizeof(int);
	v.resize(num_elements, 0);
	for (int i = 0; i < num_elements; i++)
		v[i] = i;
}

void MallocMatrix(const int M, const int N, const int K, int& lda, int& ldb, int& ldc, float*& A, float*& B, float*& C, float*& REF)
{
	A = (float*)_aligned_malloc(sizeof(float) * M * K, 32);
	B = (float*)_aligned_malloc(sizeof(float) * K * N, 32);
	lda = K;
	ldb = N;
	ldc = ((N - 1) / GEMM_NR + 1) * GEMM_NR;
	int m = ((M - 1) / GEMM_MR + 1) * GEMM_MR;
	C = (float*)_aligned_malloc(sizeof(float) * m * ldc, 32);
	REF = (float*)_aligned_malloc(sizeof(float) * m * ldc, 32);
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

void InitABCREF(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C, float* REF)
{
	mt19937 engine(random_device{}());
	uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (int i = 0; i < M * lda; i++)
		A[i] = dist(engine);
	for (int i = 0; i < K * ldb; i++)
		B[i] = dist(engine);
	fill(C, C + M * ldc, 0);
	fill(REF, REF + M * ldc, 0);
}

void PrintABC(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	cout << "Matrix A:" << endl;
	for (int i = 0; i < M; i++)
	{
		for (int k = 0; k < K; k++)
			cout << A[i * lda + k] << " ";
		cout << endl;
	}

	cout << "Matrix B:" << endl;
	for (int k = 0; k < K; k++)
	{
		for (int j = 0; j < N; j++)
			cout << B[k * ldb + j] << " ";
		cout << endl;
	}

	cout << "Matrix C:" << endl;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
			cout << C[i * ldc + j] << " ";
		cout << endl;
	}
}

void CheckResult(const int M, const int N, const int ldc, float* C, float* REF, float tolerance)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++) 
			if (fabs(C[i * ldc + j] - REF[i * ldc + j]) > tolerance)
			{
				cout << "Error: C(" << i << ", " << j << ") = " << C[i * ldc + j] << ", but expected " << REF[i * ldc + j] << endl;
				return;
			}
	cout << "Check passed!" << endl;
	return;
}
