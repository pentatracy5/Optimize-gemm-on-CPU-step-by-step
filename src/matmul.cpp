#include <matmul.h>
#include <utils.h>
#include <Eigen/Dense> 

void MatMulREF(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_map(A, M, K);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_map(B, K, N);
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_map(C, M, N);
	C_map.noalias() += A_map * B_map;
}

void MatMul0(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++)
			for (size_t k = 0; k < K; k++)
				C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void MatMul1(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t k = 0; k < K; k++)
			for (size_t j = 0; j < N; j++)
				C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void MatMul2(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	const size_t Nblock = 3072;
	for (size_t j = 0; j < N; j += Nblock)
		for (size_t i = 0; i < M; i++)
			for (size_t k = 0; k < K; k++)
				for (size_t jx = j; jx < j + Nblock; jx++)
					C[i * N + jx] += A[i * K + k] * B[k * N + jx];
}

void MatMul3(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	const size_t Nblock = 3072;
	const size_t Kblock = 256;
	for (size_t k = 0; k < K; k += Kblock)
		for (size_t j = 0; j < N; j += Nblock)
			for (size_t i = 0; i < M; i++)
				for (size_t kx = k; kx < k + Kblock; kx++)
					for (size_t jx = j; jx < j + Nblock; jx++)
						C[i * N + jx] += A[i * K + kx] * B[kx * N + jx];
}

void MatMul4(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	const size_t Mblock = 2048;
	const size_t Nblock = 3072;
	const size_t Kblock = 160;
	for (size_t i = 0; i < M; i += Mblock)
		for (size_t k = 0; k < K; k += Kblock)
			for (size_t j = 0; j < N; j += Nblock)
				for (size_t ix = i; ix < i + Mblock; ix++)
					for (size_t kx = k; kx < k + Kblock; kx++)
						for (size_t jx = j; jx < j + Nblock; jx++)
							C[ix * N + jx] += A[ix * K + kx] * B[kx * N + jx];
}

void MatMul5(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	const size_t NC = 4096;
	const size_t MC = 1024;
	const size_t KC = 320;
	const size_t NR = 16;
	const size_t MR = 4;
	for (size_t jc = 0; jc < N; jc += NC)
	{
		for (size_t ic = 0; ic < M; ic += MC)
		{
			for (size_t pc = 0; pc < K; pc += KC)
			{
				//Macro Kernel
				for (size_t jr = jc; jr < jc + NC; jr += NR)
				{
					for (size_t ir = ic; ir < ic + MC; ir += MR)
					{
						//Micro Kernel
						for (size_t kr = pc; kr < pc + KC; kr++)
						{
							for (size_t i = ir; i < ir + MR; i++)
							{
								for (size_t j = jr; j < jr + NR; j++)
								{
									C[i * N + j] += A[i * K + kr] * B[kr * N + j];
								}
							}
						}
					}
				}
			}
		}
	}
}

void PackA(size_t K, size_t MC, size_t KC, size_t MR, size_t ic, size_t pc, float* A, float* APanel)
{
	for (size_t i = 0; i < MC; i++)
	{
		for (size_t k = 0; k < KC; k++)
		{
			APanel[(i / MR * KC + k) * MR + i % MR] = A[(i + ic) * K + (k + pc)];
		}
	}
}

void PackB(size_t N, size_t KC, size_t NC, size_t NR, size_t pc, size_t jc, float* B, float* BPanel)
{
	for (size_t k = 0; k < KC; k++)
	{
		for (size_t j = 0; j < NC; j++) 
		{
			BPanel[(j / NR * KC + k) * NR + j % NR] = B[(k + pc) * N + (j + jc)];
		}
	}
}

void MatMul6(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	const size_t NC = 4096;
	const size_t MC = 1024;
	const size_t KC = 320;
	const size_t NR = 16;
	const size_t MR = 4;
	float* APanel = new float[MC * KC];
	float* BPanel = new float[KC * NC];
	for (size_t jc = 0; jc < N; jc += NC)
	{
		for (size_t ic = 0; ic < M; ic += MC)
		{
			for (size_t pc = 0; pc < K; pc += KC)
			{
				PackB(N, KC, NC, NR, pc, jc, B, BPanel);
				PackA(K, MC, KC, MR, ic, pc, A, APanel);
				//Macro Kernel
				for (size_t jr = jc; jr < jc + NC; jr += NR)
				{
					for (size_t ir = ic; ir < ic + MC; ir += MR)
					{
						//Micro Kernel
						for (size_t kr = pc; kr < pc + KC; kr++)
						{
							for (size_t i = ir; i < ir + MR; i++)
							{
								for (size_t j = jr; j < jr + NR; j++)
								{
									C[i * N + j] += APanel[((i - ic) / MR * KC + (kr - pc)) * MR + (i - ic) % MR] * BPanel[((j - jc) / NR * KC + (kr - pc)) * NR + (j - jc) % NR];
								}
							}
						}
					}
				}
			}
		}
	}
}

void MatMul7(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	const size_t NC = 4096;
	const size_t MC = 1024;
	const size_t KC = 320;
	const size_t NR = 16;
	const size_t MR = 4;
	float* APanel = new float[MC * KC];
	float* BPanel = new float[KC * NC];
	for (size_t jc = 0; jc < N; jc += NC)
	{
		for (size_t pc = 0; pc < K; pc += KC)
		{
			PackB(N, KC, NC, NR, pc, jc, B, BPanel);
			for (size_t ic = 0; ic < M; ic += MC)
			{
				PackA(K, MC, KC, MR, ic, pc, A, APanel);
				//Macro Kernel
				for (size_t jr = jc; jr < jc + NC; jr += NR)
				{
					for (size_t ir = ic; ir < ic + MC; ir += MR)
					{
						//Micro Kernel
						for (size_t kr = pc; kr < pc + KC; kr++)
						{
							for (size_t i = ir; i < ir + MR; i++)
							{
								for (size_t j = jr; j < jr + NR; j++)
								{
									C[i * N + j] += APanel[((i - ic) / MR * KC + (kr - pc)) * MR + (i - ic) % MR] * BPanel[((j - jc) / NR * KC + (kr - pc)) * NR + (j - jc) % NR];
								}
							}
						}
					}
				}
			}
		}
	}
}

void MatMul8(size_t M, size_t N, size_t K, float* A, float* B, float* C)
{
	const size_t NC = 4096;
	const size_t MC = 1024;
	const size_t KC = 320;
	const size_t NR = 16;
	const size_t MR = 4;
	float* APanel = new float[MC * KC];
	float* BPanel = new float[KC * NC];
	for (size_t ic = 0; ic < M; ic += MC)
	{
		for (size_t pc = 0; pc < K; pc += KC)
		{
			PackA(K, MC, KC, MR, ic, pc, A, APanel);
			for (size_t jc = 0; jc < N; jc += NC)
			{
				PackB(N, KC, NC, NR, pc, jc, B, BPanel);
				//Macro Kernel
				for (size_t ir = ic; ir < ic + MC; ir += MR)
				{
					for (size_t jr = jc; jr < jc + NC; jr += NR)
					{
						//Micro Kernel
						for (size_t kr = pc; kr < pc + KC; kr++)
						{
							for (size_t i = ir; i < ir + MR; i++)
							{
								for (size_t j = jr; j < jr + NR; j++)
								{
									C[i * N + j] += APanel[((i - ic) / MR * KC + (kr - pc)) * MR + (i - ic) % MR] * BPanel[((j - jc) / NR * KC + (kr - pc)) * NR + (j - jc) % NR];
								}
							}
						}
					}
				}
			}
		}
	}
}

