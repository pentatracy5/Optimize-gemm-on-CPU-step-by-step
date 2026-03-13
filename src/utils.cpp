#include <utils.h>
#include <algorithm>

using std::fill;

void InitABC(int M, int N, int K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M * K; i++)
		A[i] = (float)(i % 4 + 1);
	for (size_t i = 0; i < K * N; i++)
		B[i] = (float)(i % 4 + 1);
	fill(C, C + M * N, 0);
}