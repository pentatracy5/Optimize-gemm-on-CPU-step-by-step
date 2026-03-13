#include <matmul.h>
#include <utils.h>
#include <Eigen/Dense> 

void MatMulREF(int M, int N, int K, float* A, float* B, float* C)
{
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_map(A, M, K);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_map(B, K, N);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_map(C, M, N);
	C_map.noalias() += A_map * B_map;
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

void MatMul2(int M, int N, int K, float* A, float* B, float* C)
{
	const size_t ni = 32;
	const size_t nj = 32;
	const size_t nk = 32;
	for (size_t i = 0; i < M; i += ni)
		for (size_t k = 0; k < K; k += nk)
			for (size_t j = 0; j < N; j += nj)
				for (size_t mi = i; mi < i + ni; mi++)
					for (size_t mk = k; mk < k + nk; mk++)
						for (size_t mj = j; mj < j + nj; mj++)
							C[mi * N + mj] += A[mi * K + mk] * B[mk * N + mj];
}