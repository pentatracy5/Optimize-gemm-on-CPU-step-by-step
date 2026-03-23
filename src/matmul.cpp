#include <matmul.h>
#include <utils.h>
#include <Eigen/Dense> 

void MatMulREF(int M, int N, int L, float* A, float* B, float* C)
{
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_map(A, M, L);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_map(B, L, N);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_map(C, M, N);
	C_map.noalias() += A_map * B_map;
}

void MatMul0(int M, int N, int L, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++)
			for (size_t k = 0; k < L; k++)
				C[i * N + j] += A[i * L + k] * B[k * N + j];
}

void MatMul1(int M, int N, int L, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t k = 0; k < L; k++)
			for (size_t j = 0; j < N; j++)
				C[i * N + j] += A[i * L + k] * B[k * N + j];
}

void MatMul2(int M, int N, int L, float* A, float* B, float* C)
{
	const size_t Nblock = 3072;
	for (size_t j = 0; j < N; j += Nblock)
		for (size_t i = 0; i < M; i++)
			for (size_t k = 0; k < L; k++)
				for (size_t jx = j; jx < j + Nblock; jx++)
					C[i * N + jx] += A[i * L + k] * B[k * N + jx];
}

void MatMul3(int M, int N, int L, float* A, float* B, float* C)
{
	const size_t Nblock = 3072;
	const size_t Kblock = 256;
	for (size_t k = 0; k < L; k += Kblock)
		for (size_t j = 0; j < N; j += Nblock)
			for (size_t i = 0; i < M; i++)
				for (size_t kx = k; kx < k + Kblock; kx++)
					for (size_t jx = j; jx < j + Nblock; jx++)
						C[i * N + jx] += A[i * L + kx] * B[kx * N + jx];
}

void MatMul4(int M, int N, int L, float* A, float* B, float* C)
{
	const size_t Mblock = 2048;
	const size_t Nblock = 3072;
	const size_t Kblock = 160;
	for (size_t i = 0; i < M; i += Mblock)
		for (size_t k = 0; k < L; k += Kblock)
			for (size_t j = 0; j < N; j += Nblock)
				for (size_t ix = i; ix < i + Mblock; ix++)
					for (size_t kx = k; kx < k + Kblock; kx++)
						for (size_t jx = j; jx < j + Nblock; jx++)
							C[ix * N + jx] += A[ix * L + kx] * B[kx * N + jx];
}

