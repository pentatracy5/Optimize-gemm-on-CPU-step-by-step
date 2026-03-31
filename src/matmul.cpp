#include <matmul.h>
#include <utils.h>
#include <config.h>
#include <mkl.h>
#include <vector>
#include <omp.h>
#include <immintrin.h>

using std::vector;

void PackA(const size_t K, const size_t MC, const size_t KC, const size_t MR, float* A, float* APanel)
{
	for (int i = 0; i < MC; i += MR)
	{
		float* srcA = A + i * K;
		float* dstA = APanel + i * KC;
		vector<float*> srcA_rows(MR);
		for (size_t row = 0; row < MR; row++)
		{
			srcA_rows[row] = srcA + row * K;
		}
		for (size_t k = 0; k < KC; k++)
		{
			for (size_t row = 0; row < MR; row++)
			{
				*dstA++ = *srcA_rows[row]++;
			}
		}
	}
}

void PackB(const size_t N, const size_t KC, const size_t NC, const size_t NR, float* B, float* BPanel)
{
#ifdef USE_OMP
	#pragma omp parallel for num_threads(OMP_THREADS)
#endif
	for (int j = 0; j < NC; j += NR)
	{
		float* srcB = B + j;
		float* dstB = BPanel + j * KC;
		vector<float*> srcB_cols(NR);
		for (size_t col = 0; col < NR; col++)
		{
			srcB_cols[col] = srcB + col;
		}
		for (size_t k = 0; k < KC; k++)
		{
			for (size_t col = 0; col < NR; col++)
			{
				*dstB++ = *srcB_cols[col];
				srcB_cols[col] += N;
			}
		}
	}
}

// 基于MKL的参考实现
void MatMulREF(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		M, N, K,
		1.0f, A, K, B, N,
		1.0f, C, N);
}

// 朴素实现，循环顺序ijk
void MatMul00(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t j = 0; j < N; j++)
			for (size_t k = 0; k < K; k++)
				C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// 访存更加友好的朴素实现，循环顺序ikj
// 由于ABC都是row major的，因此ikj的循环顺序对ABC的访问都是连续的
void MatMul01(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	for (size_t i = 0; i < M; i++)
		for (size_t k = 0; k < K; k++)
			for (size_t j = 0; j < N; j++)
				C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// 单层tiling，循环顺序jikj
// 按照输入矩阵的大小（M = 4096, N = 12288, K = 640），在ikj外增加一层j的tiling
// 由于i7-13700KF的L1 cache的大小为48KB（P core）/ 32KB（E core），为了保证
// 
//	for (size_t i = 0; i < M; i++)
//		for (size_t k = 0; k < K; k++)
//			for (size_t j = 0; j < N; j++)
//				C[i * N + j] += A[i * K + k] * B[k * N + j];
//
// k++的时候，C[i * N + jx]不会被替换出L1 cache，同时保证ABC尽量铺满L1 cache，
// 再考虑其他变量和指令占用的空间，以及流水线的效率，以及Nblock能否被N整除的关系，
// 我们尽量确保
// (Nblock + 1 + Nblock) * 4 < L1 cache size，
// 最终选取Nblock = 3072
void MatMul02(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t Nblock = GEMM_Nblock;
	for (size_t j = 0; j < N; j += Nblock)
		for (size_t i = 0; i < M; i++)
			for (size_t k = 0; k < K; k++)
				for (size_t jx = j; jx < j + Nblock; jx++)
					C[i * N + jx] += A[i * K + k] * B[k * N + jx];
}

// 双层tiling，循环顺序kjikj
// 按照输入矩阵的大小（M = 4096, N = 12288, K = 640），在jikj外增加一层k的tiling
// 由于i7-13700KF的L2 cache的大小为2MB（P core）/ 4MB（两个E core共享），为了保证
// 
//	for (size_t j = 0; j < N; j += Nblock)
//		for (size_t i = 0; i < M; i++)
//			for (size_t k = 0; k < K; k++)
//				for (size_t jx = j; jx < j + Nblock; jx++)
//					C[i * N + jx] += A[i * K + k] * B[k * N + jx];
//
// i++的时候，B[kx * N + jx]不会被替换出L2 cache，同时保证ABC尽量铺满L2 cache，
// 再考虑其他变量和指令占用的空间，以及流水线的效率，以及Kblock能否被K整除的关系，
// 我们尽量确保
// (Nblock + Kblock + Kblock * Nblock) * 4 < L2 cache size，
// 最终选取Kblock = 160
void MatMul03(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t Kblock = GEMM_Kblock;
	constexpr size_t Nblock = GEMM_Nblock;
	for (size_t k = 0; k < K; k += Kblock)
		for (size_t j = 0; j < N; j += Nblock)
			for (size_t i = 0; i < M; i++)
				for (size_t kx = k; kx < k + Kblock; kx++)
					for (size_t jx = j; jx < j + Nblock; jx++)
						C[i * N + jx] += A[i * K + kx] * B[kx * N + jx];
}

// 三层tiling，循环顺序ikjikj
// 按照输入矩阵的大小（M = 4096, N = 12288, K = 640），在kjikj外增加一层i的tiling
// 由于i7-13700KF的L3 cache的大小为30MB（所有core共享），为了保证
// 
//	for (size_t k = 0; k < K; k += Kblock)
//		for (size_t j = 0; j < N; j += Nblock)
//			for (size_t i = 0; i < M; i++)
//				for (size_t kx = k; kx < k + Kblock; kx++)
//					for (size_t jx = j; jx < j + Nblock; jx++)
//						C[i * N + jx] += A[i * K + kx] * B[kx * N + jx];
//
// j += Nblock的时候，A[i * K + kx]不会被替换出L3 cache，同时保证ABC尽量铺满L3 cache，
// 再考虑其他变量和指令占用的空间，以及流水线的效率，以及Mblock能否被M整除的关系，
// 我们尽量确保
// (Mblock * Nblock + Mblock * Kblock + Kblock * Nblock) * 4 < L3 cache size，
// 最终选取Mblock = 2048
void MatMul04(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t Mblock = GEMM_Mblock;
	constexpr size_t Kblock = GEMM_Kblock;
	constexpr size_t Nblock = GEMM_Nblock;
	for (size_t i = 0; i < M; i += Mblock)
		for (size_t k = 0; k < K; k += Kblock)
			for (size_t j = 0; j < N; j += Nblock)
				for (size_t ix = i; ix < i + Mblock; ix++)
					for (size_t kx = k; kx < k + Kblock; kx++)
						for (size_t jx = j; jx < j + Nblock; jx++)
							C[ix * N + jx] += A[ix * K + kx] * B[kx * N + jx];
}

// 三层tiling，循环顺序ikjikj，并且使用intrinsic指令进行向量化
void MatMul05(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t Mblock = GEMM_Mblock;
	constexpr size_t Kblock = GEMM_Kblock;
	constexpr size_t Nblock = GEMM_Nblock;
	for (size_t i = 0; i < M; i += Mblock)
		for (size_t k = 0; k < K; k += Kblock)
			for (size_t j = 0; j < N; j += Nblock)
				for (size_t ix = i; ix < i + Mblock; ix++)
					for (size_t kx = k; kx < k + Kblock; kx++)
					{
						__m256 a = _mm256_set1_ps(A[ix * K + kx]);
						for (size_t jx = j; jx < j + Nblock; jx += 8)
						{
							__m256 b = _mm256_load_ps(B + kx * N + jx);
							__m256 c = _mm256_load_ps(C + ix * N + jx);
							c = _mm256_fmadd_ps(a, b, c);
							_mm256_store_ps(C + ix * N + jx, c);
						}
					}
}

// GoToBLAS风格的tiling，循环顺序jikjik，并且使用intrinsic指令进行向量化，未pack
// 由于i7-13700KF支持的向量化指令集AVX2有16个ymm寄存器，而
//						__m256 a = _mm256_set1_ps(A[ix * K + kx]);
//						for (size_t jx = j; jx < j + Nblock; jx += 8)
//						{
//							__m256 b = _mm256_load_ps(B + kx * N + jx);
//							__m256 c = _mm256_load_ps(C + ix * N + jx);
//							c = _mm256_fmadd_ps(a, b, c);
//							_mm256_store_ps(C + ix * N + jx, c);
//						}
// 仅使用了3个寄存器，利用率不高，因此考虑设计一个算法，使得每次能利用更多的寄存器进行存储和向量化计算。
// 若每次计算 
// C（4 * 16）= A（4 * 1）* B（1 * 16），
// 则每次可使用8个寄存器存储C，1个寄存器存储A，2个寄存器存储B，总共使用11个寄存器，利用率较高。
// 配合这样的利用寄存器进行矩阵计算的方式，我们需要重新设计tiling的顺序和尺寸。
// 首先，由于此时C的尺寸大于A和B，因此我们可以在外层套上k的循环，便能复用寄存器中C的数据。
// 我们将
// 
//	for k in range(KC)：
//		C（MR * NR）= A_k（MR * 1）* B_k（1 * NR）
// 
// 称为Micro Kernel层次，其中MR = 4，NR = 16。
// 为了使Micro Kernel中的A和B能铺满L1 cache，KC大概率是一个远大于MR和NR的数，
// 因此对于整个循环来说，B（KC * NR）的尺寸要大于A（MR * KC）和C（MR * NR）。
// 为了尽可能地复用B的tile，我们在外层套再上i的循环，同时保证ABC尽量铺满L1 cache，
// 再考虑其他变量和指令占用的空间，以及流水线的效率，以及KC能否被K整除的关系，我们尽量确保
// 2 * MR * NR + 2 * MR * KC + KC * NR < L1 cache size，
// 最终选取KC = 320。
// 类似的，对于
// 
//	for i in range(MC) step MR：
//		for k in range(KC)：
//			C（MR * NR）= A_k（MR * 1）* B_k（1 * NR）
// 
// 的循环，为了保证ABC尽量铺满L2 cache，我们大概率会取比较大的MC，从而A的tile的尺寸最大，
// 为了复用尺寸最大的A的tile，我们可以在外层套上j的循环，
// 再考虑其他变量和指令占用的空间，以及流水线的效率，以及MC能否被M整除的关系，我们尽量确保
// 2 * MC * NR + MC * KC + 2 * KC * NR < L2 cache size，
// 最终选取MC = 1024。
// 同理，对于
// 
//	for j in range(NC) step NR：
//		for i in range(MC) step MR：
//			for k in range(KC)：
//				C（MR * NR）= A_k（MR * 1）* B_k（1 * NR）
// 
// 的循环，为了保证ABC尽量铺满L3 cache，我们大概率会取比较大的NC，从而C的tile的尺寸最大，
// 为了复用尺寸最大的C的tile，因此在外层套上k的循环，同时
// 再考虑其他变量和指令占用的空间，以及流水线的效率，以及NC能否被N整除的关系，我们尽量确保
// MC * NC + 2 * MC * KC + 2 * KC * NC < L3 cache size，
// 最终选取NC = 4096。
// 最后，在
// 
//	for k in range(K) step KC：
//		for j in range(NC) step NR：
//			for i in range(MC) step MR：
//				for k in range(KC)：
//					C（MR * NR）= A_k（MR * 1）* B_k（1 * NR）
// 
// 的循环之外，我们再套上一个j的循环和一个i的循环，完成对整个矩阵的计算。
void MatMul06(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t NC = GEMM_NC;
	constexpr size_t MC = GEMM_MC;
	constexpr size_t KC = GEMM_KC;
	constexpr size_t NR = GEMM_NR;
	constexpr size_t MR = GEMM_MR;
	constexpr size_t nr = NR / 8;
	float* macroA, * macroB, * macroC;
	float* microA, * microB, * microC;
	for (size_t jc = 0; jc < N; jc += NC)
	{
		for (size_t ic = 0; ic < M; ic += MC)
		{
			macroC = C + ic * N + jc;
			for (size_t pc = 0; pc < K; pc += KC)
			{
				macroA = A + ic * K + pc;
				macroB = B + pc * N + jc;
				//Macro Kernel
				for (size_t jr = 0; jr < NC; jr += NR)
				{
					microB = macroB + jr;
					for (size_t ir = 0; ir < MC; ir += MR)
					{
						microA = macroA + ir * K;
						microC = macroC + ir * N + jr;
						//Micro Kernel
						__m256 c[MR][nr];
						for (size_t i = 0; i < MR; i++)
							for (size_t l = 0; l < nr; l++)
								c[i][l] = _mm256_load_ps(microC + i * N + l * 8);
						__m256 b[nr];
						for (size_t kr = 0; kr < KC; kr++)
						{
							for (size_t l = 0; l < nr; l++)
								b[l] = _mm256_load_ps(microB + kr * N + l * 8);

							for (size_t i = 0; i < MR; i++)
							{
								__m256 a = _mm256_set1_ps(microA[i * K + kr]);
								for (size_t l = 0; l < nr; l++)
									c[i][l] = _mm256_fmadd_ps(a, b[l], c[i][l]);
							}
						}
						for (size_t i = 0; i < MR; i++)
							for (size_t l = 0; l < nr; l++)
								_mm256_store_ps(microC + i * N + l * 8, c[i][l]);
					}
				}
			}
		}
	}
}

// GoToBLAS风格的tiling，循环顺序jikjik，并且使用intrinsic指令进行向量化，pack
// 在MatMul6的基础上增加了对A和B的pack，使得内存访问更加连续
// pack的时机是 jik -> packA -> packB -> jik，这里将pack之后的循环过程称为Macro Kernel层次
// 时机的选取并没有经过验证
void MatMul07(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t NC = GEMM_NC;
	constexpr size_t MC = GEMM_MC;
	constexpr size_t KC = GEMM_KC;
	constexpr size_t NR = GEMM_NR;
	constexpr size_t MR = GEMM_MR;
	constexpr size_t nr = NR / 8;
	float* macroA, * macroB, * macroC;
	float* microA, * microB, * microC;
	float* APanel = (float*)_aligned_malloc(sizeof(float) * MC * KC, 32);
	float* BPanel = (float*)_aligned_malloc(sizeof(float) * KC * NC, 32);
	for (size_t jc = 0; jc < N; jc += NC)
	{
		for (size_t ic = 0; ic < M; ic += MC)
		{
			macroC = C + ic * N + jc;
			for (size_t pc = 0; pc < K; pc += KC)
			{
				macroA = A + ic * K + pc;
				macroB = B + pc * N + jc;
				PackA(K, MC, KC, MR, macroA, APanel);
				PackB(N, KC, NC, NR, macroB, BPanel);
				//Macro Kernel
				for (size_t jr = 0; jr < NC; jr += NR)
				{
					microB = BPanel + jr * KC;
					for (size_t ir = 0; ir < MC; ir += MR)
					{
						microA = APanel + ir * KC;
						microC = macroC + ir * N + jr; 
						//Micro Kernel
						__m256 c[MR][nr];
						for (size_t i = 0; i < MR; i++)
							for (size_t l = 0; l < nr; l++)
								c[i][l] = _mm256_load_ps(microC + i * N + l * 8);
						__m256 b[nr];
						for (size_t kr = 0; kr < KC; kr++)
						{
							for (size_t l = 0; l < nr; l++)
								b[l] = _mm256_load_ps(microB + kr * NR + l * 8);

							for (size_t i = 0; i < MR; i++)
							{
								__m256 a = _mm256_set1_ps(microA[kr * MR + i]);
								for (size_t l = 0; l < nr; l++)
									c[i][l] = _mm256_fmadd_ps(a, b[l], c[i][l]);
							}
						}
						for (size_t i = 0; i < MR; i++)
							for (size_t l = 0; l < nr; l++)
								_mm256_store_ps(microC + i * N + l * 8, c[i][l]);
					}
				}
			}
		}
	}
	_aligned_free(APanel);
	_aligned_free(BPanel);
}

// GoToBLAS风格的tiling，循环顺序jkijik，并且使用intrinsic指令进行向量化，pack
// 在MatMul7的基础上调整了循环以及pack的顺序，节省了packB的次数，
// tradeoff是从复用C（MC * NC）变为复用B（KC * NC），复用的tile的尺寸变小
// pack的时机是 jk -> packB -> i -> packA -> jik
void MatMul08(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t NC = GEMM_NC;
	constexpr size_t MC = GEMM_MC;
	constexpr size_t KC = GEMM_KC;
	constexpr size_t NR = GEMM_NR;
	constexpr size_t MR = GEMM_MR;
	constexpr size_t nr = NR / 8;
	float* macroA, * macroB, * macroC;
	float* microA, * microB, * microC;
	float* APanel = (float*)_aligned_malloc(sizeof(float) * MC * KC, 32);
	float* BPanel = (float*)_aligned_malloc(sizeof(float) * KC * NC, 32);
	for (size_t jc = 0; jc < N; jc += NC)
	{
		for (size_t pc = 0; pc < K; pc += KC)
		{
			macroB = B + pc * N + jc;
			PackB(N, KC, NC, NR, macroB, BPanel);
			for (size_t ic = 0; ic < M; ic += MC)
			{
				macroA = A + ic * K + pc;
				macroC = C + ic * N + jc;
				PackA(K, MC, KC, MR, macroA, APanel);
				//Macro Kernel
				for (size_t jr = 0; jr < NC; jr += NR)
				{
					microB = BPanel + jr * KC;
					for (size_t ir = 0; ir < MC; ir += MR)
					{
						microA = APanel + ir * KC;
						microC = macroC + ir * N + jr;
						//Micro Kernel
						__m256 c[MR][nr];
						for (size_t i = 0; i < MR; i++)
							for (size_t l = 0; l < nr; l++)
								c[i][l] = _mm256_load_ps(microC + i * N + l * 8);
						__m256 b[nr];
						for (size_t kr = 0; kr < KC; kr++)
						{
							for (size_t l = 0; l < nr; l++)
								b[l] = _mm256_load_ps(microB + kr * NR + l * 8);

							for (size_t i = 0; i < MR; i++)
							{
								__m256 a = _mm256_set1_ps(microA[kr * MR + i]);
								for (size_t l = 0; l < nr; l++)
									c[i][l] = _mm256_fmadd_ps(a, b[l], c[i][l]);
							}
						}
						for (size_t i = 0; i < MR; i++)
							for (size_t l = 0; l < nr; l++)
								_mm256_store_ps(microC + i * N + l * 8, c[i][l]);
					}
				}
			}
		}
	}
	_aligned_free(APanel);
	_aligned_free(BPanel);
}

// GoToBLAS风格的tiling，循环顺序jkijik，并且使用intrinsic指令进行向量化，pack，启用OpenMP
// OpenMP在两处使用，第一处是packB的循环，第二处是packB之后的循环
// 因为packB之后的循环包含了packA，所以在分配APanel时，需要考虑线程数，分配MC * KC * OMP_THREADS的空间
// 从硬件层面来说，i7-13700KF的不同核心之间的L2 cache不共享（除了每两个E core共享4MB L2 cache的情况）
// 而算法设计的目的就是让APanel铺满L2 cache，所以我们选择packA的时机开启OpenMP多线程
// 在这之前，我们对packB的过程开启多线程，也能提高硬件资源的利用率
void MatMul09(const size_t M, const size_t N, const size_t K, float* A, float* B, float* C)
{
	constexpr size_t NC = GEMM_NC_OMP;
	constexpr size_t MC = GEMM_MC;
	constexpr size_t KC = GEMM_KC;
	constexpr size_t NR = GEMM_NR;
	constexpr size_t MR = GEMM_MR;
	constexpr size_t nr = NR / 8;
	float* macroB;
	float* APanel = (float*)_aligned_malloc(sizeof(float) * MC * KC * OMP_THREADS, 32);
	float* BPanel = (float*)_aligned_malloc(sizeof(float) * KC * NC, 32);
	for (size_t jc = 0; jc < N; jc += NC)
	{
		for (size_t pc = 0; pc < K; pc += KC)
		{
			macroB = B + pc * N + jc;
			PackB(N, KC, NC, NR, macroB, BPanel);

			#pragma omp parallel num_threads(OMP_THREADS)
			{
				float* macroA, * macroC;
				float* microA, * microB, * microC;
				size_t range = M / omp_get_num_threads();
				size_t tid = omp_get_thread_num();
				size_t start = tid * range;
				size_t end = start + range;
				for (size_t ic = start; ic < end; ic += MC)
				{
					macroA = A + ic * K + pc;
					macroC = C + ic * N + jc;
					PackA(K, MC, KC, MR, macroA, APanel + tid * MC * KC);
					//Macro Kernel
					for (size_t jr = 0; jr < NC; jr += NR)
					{
						microB = BPanel + jr * KC;
						for (size_t ir = 0; ir < MC; ir += MR)
						{
							microA = APanel + tid * MC * KC + ir * KC;
							microC = macroC + ir * N + jr;
							//Micro Kernel
							__m256 c[MR][nr];
							for (size_t i = 0; i < MR; i++)
								for (size_t l = 0; l < nr; l++)
									c[i][l] = _mm256_load_ps(microC + i * N + l * 8);
							__m256 b[nr];
							for (size_t kr = 0; kr < KC; kr++)
							{
								for (size_t l = 0; l < nr; l++)
									b[l] = _mm256_load_ps(microB + kr * NR + l * 8);

								for (size_t i = 0; i < MR; i++)
								{
									__m256 a = _mm256_set1_ps(microA[kr * MR + i]);
									for (size_t l = 0; l < nr; l++)
										c[i][l] = _mm256_fmadd_ps(a, b[l], c[i][l]);
								}
							}
							for (size_t i = 0; i < MR; i++)
								for (size_t l = 0; l < nr; l++)
									_mm256_store_ps(microC + i * N + l * 8, c[i][l]);
						}
					}
				}
			}
		}
	}
	_aligned_free(APanel);
	_aligned_free(BPanel);
}