#include <config.h>
#include <utils.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

using std::fill;
using std::cout;
using std::endl;
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;
using std::vector;

void* AlignedMalloc(size_t size, size_t alignment)
{
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
    {
        return nullptr;
    }
    return ptr;
#endif
}

void AlignedFree(void* ptr)
{
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

vector<int> v;
void ClearCache()
{
	constexpr int cache_size = 32 * 1024 * 1024;
	constexpr int num_elements = cache_size / sizeof(int);
	v.resize(num_elements, 0);
	volatile int* data = const_cast<volatile int*>(v.data());
	for (int i = 0; i < num_elements; ++i)
	{
		data[i] = 0;
	}
	std::atomic_thread_fence(std::memory_order_seq_cst);
}

void MallocMatrix(const int M, const int N, const int K, int& lda, int& ldb, int& ldc, float*& A, float*& B, float*& C, float*& REF)
{
	A = (float*)AlignedMalloc(sizeof(float) * M * K, 32);
	B = (float*)AlignedMalloc(sizeof(float) * K * N, 32);
	lda = K;
	ldb = N;
	ldc = ((N - 1) / GEMM_NR + 1) * GEMM_NR;
	int m = ((M - 1) / GEMM_MR + 1) * GEMM_MR;
	C = (float*)AlignedMalloc(sizeof(float) * m * ldc, 32);
	REF = (float*)AlignedMalloc(sizeof(float) * m * ldc, 32);
}

void FreeMatrix(float*& A, float*& B, float*& C, float*& REF)
{
	AlignedFree(A);
	AlignedFree(B);
	AlignedFree(C);
	AlignedFree(REF);
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
	int m = ((M - 1) / GEMM_MR + 1) * GEMM_MR;
	fill(C, C + m * ldc, 0.0f);
	fill(REF, REF + m * ldc, 0.0f);
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
	return;
}
