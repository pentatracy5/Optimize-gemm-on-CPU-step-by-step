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
	constexpr size_t cache_size = 32 * 1024 * 1024;
	constexpr size_t num_elements = cache_size / sizeof(int);
	v.resize(num_elements, 0);
	for (size_t i = 0; i < num_elements; i++)
		v[i] = i;
}

void MallocMatrix(const size_t M, const size_t N, const size_t K, size_t& lda, size_t& ldb, size_t& ldc, float*& A, float*& B, float*& C, float*& REF)
{
	A = (float*)_aligned_malloc(sizeof(float) * M * K, 32);
	B = (float*)_aligned_malloc(sizeof(float) * K * N, 32);
	lda = K;
	ldb = N;
	ldc = ((N - 1) / GEMM_NR + 1) * GEMM_NR;
	size_t m = ((M - 1) / GEMM_MR + 1) * GEMM_MR;
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

void InitABCREF(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C, float* REF)
{
	mt19937 engine(random_device{}());
	uniform_real_distribution<float> dist(0.0f, 1.0f);

	for (size_t i = 0; i < M * lda; i++)
		A[i] = dist(engine);
	for (size_t i = 0; i < K * ldb; i++)
		B[i] = dist(engine);
	fill(C, C + M * ldc, 0);
	fill(REF, REF + M * ldc, 0);
}

void PrintABC(const size_t M, const size_t N, const size_t K, const size_t lda, const size_t ldb, const size_t ldc, float* A, float* B, float* C)
{
	cout << "Matrix A:" << endl;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t k = 0; k < K; k++)
			cout << A[i * lda + k] << " ";
		cout << endl;
	}

	cout << "Matrix B:" << endl;
	for (size_t k = 0; k < K; k++)
	{
		for (size_t j = 0; j < N; j++)
			cout << B[k * ldb + j] << " ";
		cout << endl;
	}

	cout << "Matrix C:" << endl;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
			cout << C[i * ldc + j] << " ";
		cout << endl;
	}
}

void CheckResult(const size_t M, const size_t N, const size_t ldc, float* C, float* REF, float tolerance)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++) 
			if (fabs(C[i * ldc + j] - REF[i * ldc + j]) > tolerance)
			{
				cout << "Error: C(" << i << ", " << j << ") = " << C[i * ldc + j] << ", but expected " << REF[i * ldc + j] << endl;
				return;
			}
	cout << "Check passed!" << endl;
	return;
}
