#include <matmul.h>
#include <utils.h>
#include <Eigen/Dense> 

void MatMulGT(int M, int N, int K, float* A, float* B, float* C)
{
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_map(A, M, K);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_map(B, K, N);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_map(C, M, N);
	C_map.noalias() = A_map * B_map;
}

void MatMul0(int M, int N, int K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++)
			for (size_t k = 0; k < K; k++)
				C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void MatMul1(int M, int N, int K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t k = 0; k < K; k++)
			for (size_t j = 0; j < N; j++)
				C[i * N + j] += A[i * K + k] * B[k * N + j];
}