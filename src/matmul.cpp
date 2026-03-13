#include <matmul.h>
#include <utils.h>

void MatMul0(int M, int N, int K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++)
			for (size_t k = 0; k < K; k++)
				C(i, j) += A(i, k) * B(k, j);
}