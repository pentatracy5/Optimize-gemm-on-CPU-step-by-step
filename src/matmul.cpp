#include <matmul.h>
#include <utils.h>
#include <config.h>
#include <mkl.h>
#include <vector>
#include <omp.h>
#include <immintrin.h>

using std::vector;
using std::min;

/**
 * @brief 将矩阵 A 的一个子块打包到连续缓冲区 APanel 中。
 *
 * 该函数用于高性能矩阵乘法（如 BLIS/GotoBLAS 风格）的数据重排阶段。
 * 它从行优先存储的矩阵 A 中提取一个大小为 MC × KC 的子矩阵（列起始为 0），
 * 按 MR 行划分为多个行块，每个行块内按 **列主序** 连续存储到 APanel 中，
 * 以提升后续微内核计算时的缓存局部性。
 *
 * @param lda    矩阵 A 的 leading dimension（行优先，即一行中的元素个数，通常 ≥ KC）
 * @param MC     子矩阵的总行数
 * @param KC     子矩阵的总列数
 * @param MR     每个行块的最大行数（通常为微内核的行累加尺寸）
 * @param A      源矩阵指针，指向子矩阵的起始位置（列偏移为 0）
 * @param APanel 目标打包缓冲区指针，其大小至少为 MC * KC 个 float
 *
 * @details 算法流程：
 *   - 外层循环遍历行块 i = 0, MR, 2MR, ...，直到覆盖 MC 行。
 *   - 当前块实际行数 currentMR = min(MR, MC - i)。
 *   - 对于每个块，先初始化一个长度为 MR 的行指针数组 srcA_rows：
 *       前 currentMR 个指针指向块内对应行的起始地址（srcA + row * lda）；
 *       若 currentMR < MR，则剩余指针重复指向块内的 **第一行**（srcA）。
 *   - 然后内层双重循环：逐列 k = 0..KC-1，逐行 row = 0..MR-1，
 *       依次将 srcA_rows[row][k] 复制到 dstA，并移动各个行指针。
 *   - 结果：每个行块在 APanel 中占据连续 KC * MR 个 float，
 *       存储顺序为 (col0_row0, col0_row1, ..., col0_row_{MR-1},
 *                     col1_row0, ..., col_{KC-1}_row_{MR-1})。
 *
 * @note 当最后一个行块不足 MR 行时，多余的行位置会重复打包第一行的数据，
 *       调用方必须确保后续计算只使用前 currentMR 行，忽略填充部分。
 *
 * @warning 函数假设 A 以行优先存储，且子矩阵的列索引从 0 开始。
 *          如果实际子矩阵列偏移不为 0，调用前需将 A 指针调整到正确位置。
 */
void PackA(const int lda, const int MC, const int KC, const int MR, float* A, float* APanel)
{
	for (int i = 0; i < MC; i += MR)
	{
		int currentMR = min(MR, MC - i);
		float* srcA = A + i * lda;
		float* dstA = APanel + i * KC;
		vector<float*> srcA_rows(MR);
		for (int row = 0; row < currentMR; row++)
		{
			srcA_rows[row] = srcA + row * lda;
		}
		for (int col = currentMR; col < MR; col++)
		{
			srcA_rows[col] = srcA;
		}
		for (int k = 0; k < KC; k++)
		{
			for (int row = 0; row < MR; row++)
			{
				*dstA++ = *srcA_rows[row]++;
			}
		}
	}
}

/**
 * @brief 将矩阵 B 的一个子块打包到连续缓冲区 BPanel 中。
 *
 * 该函数用于高性能矩阵乘法（如 BLIS/GotoBLAS 风格）的数据重排阶段。
 * 它从 **列优先** 存储的矩阵 B 中提取一个大小为 KC × NC 的子矩阵（行起始为 0），
 * 按 NR 列划分为多个列块，每个列块内按 **行主序** 连续存储到 BPanel 中，
 * 以提升后续微内核计算时的缓存局部性。
 *
 * @param ldb    矩阵 B 的 leading dimension（列优先，即一列中的元素个数，通常 ≥ KC）
 * @param KC     子矩阵的总行数
 * @param NC     子矩阵的总列数
 * @param NR     每个列块的最大列数（通常为微内核的列累加尺寸）
 * @param B      源矩阵指针，指向子矩阵的起始位置（行偏移为 0）
 * @param BPanel 目标打包缓冲区指针，其大小至少为 KC * NC 个 float
 *
 * @details 算法流程：
 *   - 外层循环遍历列块 j = 0, NR, 2NR, ...，直到覆盖 NC 列。
 *   - 当前块实际列数 currentNR = min(NR, NC - j)。
 *   - 对于每个块，先初始化一个长度为 NR 的列指针数组 srcB_cols：
 *       前 currentNR 个指针指向块内对应列的首元素地址（srcB + col）；
 *       若 currentNR < NR，则剩余指针重复指向块内的 **第一列**（srcB）。
 *   - 然后内层双重循环：逐行 k = 0..KC-1，逐列 col = 0..NR-1，
 *       依次将 srcB_cols[col] 指向的元素复制到 dstB，
 *       然后该列指针下移一行（加上 ldb）。
 *   - 结果：每个列块在 BPanel 中占据连续 KC * NR 个 float，
 *       存储顺序为 (row0_col0, row0_col1, ..., row0_col_{NR-1},
 *                     row1_col0, ..., row_{KC-1}_col_{NR-1})。
 *
 * @note 当最后一个列块不足 NR 列时，多余的列位置会重复打包第一列的数据，
 *       调用方必须确保后续计算只使用前 currentNR 列，忽略填充部分。
 *
 * @warning 函数假设矩阵 B 以 **列优先** 存储（Fortran 风格），
 *          且子矩阵的行索引从 0 开始。如果实际子矩阵行偏移不为 0，
 *          调用前需将 B 指针调整到正确位置。
 *
 * @note 若启用 OpenMP（定义 USE_OMP），外层循环会被并行化，
 *       每个线程处理一组列块，从而加速大数据量打包。
 */
void PackB(const int ldb, const int KC, const int NC, const int NR, float* B, float* BPanel)
{
	#pragma omp parallel for num_threads(OMP_THREADS)
	for (int j = 0; j < NC; j += NR)
	{
		int currentNR = min(NR, NC - j);
		float* srcB = B + j;
		float* dstB = BPanel + j * KC;
		vector<float*> srcB_cols(NR);
		for (int col = 0; col < currentNR; col++)
		{
			srcB_cols[col] = srcB + col;
		}
		for (int col = currentNR; col < NR; col++)
		{
			srcB_cols[col] = srcB;
		}
		for (int k = 0; k < KC; k++)
		{
			for (int col = 0; col < NR; col++)
			{
				*dstB++ = *srcB_cols[col];
				srcB_cols[col] += ldb;
			}
		}
	}
}

/**
 * @brief 矩阵乘法的参考实现（直接调用 BLAS 库的 cblas_sgemm）。
 *
 * 该函数作为基准参考，计算 C = A * B + C
 */
void MatMulREF(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		M, N, K,
		1.0f, A, lda, B, ldb,
		1.0f, C, ldc);
}

/**
 * @brief 朴素的三重循环矩阵乘法（无任何优化）。
 *
 * 该函数直接按定义计算 C = A * B + C
 */
void MatMul00(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < K; k++)
				C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
}

/**
 * @brief 改进版矩阵乘法：交换循环顺序，提升访存连续性。
 *
 * 与 MatMul00 相比，该版本将 j 循环移至最内层，顺序为 i -> k -> j。
 * 这一调整带来了显著的性能提升：
 *   - 内层 j 循环访问 B[k * ldb + j] 和 C[i * ldc + j] 时，地址连续递增，
 *     充分利用了空间局部性和 CPU 缓存行的预取特性。
 */
void MatMul01(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	for (int i = 0; i < M; i++)
		for (int k = 0; k < K; k++)
			for (int j = 0; j < N; j++)
				C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
}

/**
 * @brief 在 MatMul01 基础上引入 N 维分块（blocking），进一步优化缓存利用。
 *
 * 相比 MatMul01 的改进：
 *   - MatMul01 内层 j 循环会遍历整行的 C 的数据，当 N 很大时，导致缓存容量不足，频繁发生缓存缺失。
 *   - MatMul02 将 N 方向划分为大小为 Nblock 的块，外层先固定一个列块（j 到 j+currentNblock），
 *     然后对于固定的 i ，在内层 k-jx 两重循环中， C 的子块能够驻留在 L1 缓存。
 * 
 * @note Nblock 的选择应考虑 L1 缓存大小，以确保数据能够驻留在 L1 中，在这个前提下尽量铺满 L1。
 */
void MatMul02(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int Nblock = GEMM_NC;
	int currentNblock;
	for (int j = 0; j < N; j += Nblock)
	{
		currentNblock = min(Nblock, N - j);
		for (int i = 0; i < M; i++)
			for (int k = 0; k < K; k++)
				for (int jx = j; jx < j + currentNblock; jx++)
					C[i * ldc + jx] += A[i * lda + k] * B[k * ldb + jx];
	}
}

/**
 * @brief 在 MatMul02 基础上引入 K 维分块（blocking），进一步优化缓存利用。
 *
 * 相比 MatMul02 的改进：
 *   - MatMul02 内层 k-jx 循环会遍历整列的 B 子块，当 K 很大时，导致缓存容量不足，频繁发生缓存缺失。
 *   - MatMul03 将 K 方向划分为大小为 Kblock 的块，外层先固定一个 A 列块和 B 行块（k 到 k+currentKblock），
 *     然后对于固定的列块（j 到 j+currentNblock），内层 i-kx-jx 三重循环中，
 *	    B 的子块能够驻留在 L2 缓存中。
 * 
 * @note Kblock 的选择应考虑 L2 缓存大小，以确保数据能够驻留在 L2 中，在这个前提下尽量铺满 L2。
 */
void MatMul03(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int Kblock = GEMM_KC;
	constexpr int Nblock = GEMM_NC;
	int currentKblock, currentNblock;
	for (int k = 0; k < K; k += Kblock)
	{
		currentKblock = min(Kblock, K - k);
		for (int j = 0; j < N; j += Nblock)
		{
			currentNblock = min(Nblock, N - j);
			for (int i = 0; i < M; i++)
				for (int kx = k; kx < k + currentKblock; kx++)
					for (int jx = j; jx < j + currentNblock; jx++)
						C[i * ldc + jx] += A[i * lda + kx] * B[kx * ldb + jx];
		}
	}
}

/**
 * @brief 在 MatMul03 基础上引入 M 维分块（blocking），进一步优化缓存利用。
 *
 * 相比 MatMul03 的改进：
 *   - MatMul03 内层 i-kx-jx 循环会遍历整行的 A 子块，当 M 很大时，导致缓存容量不足，频繁发生缓存缺失。
 *   - MatMul04 将 M 方向划分为大小为 Mblock 的块，外层先固定一个行块（i 到 i+currentMblock），
 *     然后对于固定的列块（k 到 k+currentKblock），内层 j-ix-kx-jx 四重循环中，
 *	    A 的子块能够驻留在 L3 缓存中。
 * 
 * @note Mblock 的选择应考虑 L3 缓存大小，以确保数据能够驻留在 L3 中，在这个前提下尽量铺满 L3。
 */
void MatMul04(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int Mblock = GEMM_MC;
	constexpr int Kblock = GEMM_KC;
	constexpr int Nblock = GEMM_NC;
	int currentMblock, currentKblock, currentNblock;
	for (int i = 0; i < M; i += Mblock)
	{
		currentMblock = min(Mblock, M - i);
		for (int k = 0; k < K; k += Kblock)
		{
			currentKblock = min(Kblock, K - k);
			for (int j = 0; j < N; j += Nblock)
			{
				currentNblock = min(Nblock, N - j);
				for (int ix = i; ix < i + currentMblock; ix++)
					for (int kx = k; kx < k + currentKblock; kx++)
						for (int jx = j; jx < j + currentNblock; jx++)
							C[ix * ldc + jx] += A[ix * lda + kx] * B[kx * ldb + jx];
			}
		}
	}
}

/**
 * @brief 在 MatMul04 基础上引入 AVX2 向量化，利用 SIMD 指令并行计算。
 *
 * 相比 MatMul04 的改进：
 *   - MatMul04 最内层循环（jx）逐个浮点累乘，每个时钟周期只能处理 1 个元素。
 *   - MatMul05 使用 AVX2 指令集（256 位寄存器），一次加载 8 个浮点数（__m256），
 *     结合 FMA（融合乘加）指令 _mm256_fmadd_ps，一个时钟周期可完成 8 对乘加运算。
 *   - 内层循环按 8 步长展开，并用 _mm256_load_ps / _mm256_store_ps 对齐访存，
 *     大幅提升浮点运算吞吐量和内存带宽利用率。
 *   - 对尾部不足 8 个元素的列块，使用掩码加载/存储指令（_mm256_maskload_ps / _mm256_maskstore_ps）
 *     安全处理剩余部分，避免越界或对齐错误。
 *
 * @note 要求 Nblock 为 8 的倍数以获得最佳性能，矩阵 B 和 C 的起始地址需 32 字节对齐。
 */
void MatMul05(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int Mblock = GEMM_MC;
	constexpr int Kblock = GEMM_KC;
	constexpr int Nblock = GEMM_NC;
	int currentMblock, currentKblock, currentNblock;
	for (int i = 0; i < M; i += Mblock)
	{
		currentMblock = min(Mblock, M - i);
		for (int k = 0; k < K; k += Kblock)
		{
			currentKblock = min(Kblock, K - k);
			for (int j = 0; j < N; j += Nblock)
			{
				currentNblock = min(Nblock, N - j);
				for (int ix = i; ix < i + currentMblock; ix++)
					for (int kx = k; kx < k + currentKblock; kx++)
					{
						__m256 a = _mm256_set1_ps(A[ix * lda + kx]);
						int jx;
						for (jx = j; jx <= j + currentNblock - 8; jx += 8)
						{
							__m256 b = _mm256_load_ps(B + kx * ldb + jx);
							__m256 c = _mm256_load_ps(C + ix * ldc + jx);
							c = _mm256_fmadd_ps(a, b, c);
							_mm256_store_ps(C + ix * ldc + jx, c);
						}
						if (j + currentNblock - jx > 0)
						{
							alignas(32) int mask_array[8]{ 0 };
							for (int l = 0; l < j + currentNblock - jx; ++l)
								mask_array[l] = -1;
							__m256i mask = _mm256_load_si256((const __m256i*)mask_array);

							__m256 b = _mm256_maskload_ps(B + kx * ldb + jx, mask);
							__m256 c = _mm256_maskload_ps(C + ix * ldc + jx, mask);
							c = _mm256_fmadd_ps(a, b, c);
							_mm256_maskstore_ps(C + ix * ldc + jx, mask, c);
						}
					}
			}
		}
	}
}

/**
 * @brief 在 MatMul05 基础上引入多级分块与寄存器级累加（Micro Kernel）。
 *
 * 相比 MatMul05 的改进：
 *   - **多级分块结构**：使用 NC/MC/KC 作为外层分块（适配 L3 缓存），内层再用 NR/MR 分块（适配寄存器/L1）。
 *     比 MatMul05 的单层分块（Mblock/Kblock/Nblock）更好地利用了缓存层次。
 *   - **寄存器块累加（c[MR][nr]）**：在 Micro Kernel 中，将 C 的 MR×NR 子块预加载到 __m256 寄存器数组，
 *     然后在 K 维度内循环累加乘积累，最后才写回内存。而 MatMul05 每次 jx 迭代都需加载/存储 C，
 *     导致大量冗余访存。
 *   - **显式 Micro Kernel 分离**：将最内层计算封装为 Micro Kernel，便于独立优化（如循环展开、指令调度），
 *     也为后续手工汇编或预取等高级优化奠定基础。
 *
 * @note 循环顺序的设计以及NC/MC/KC/MR/NR的选取
 * 
 *   - 每次在外层套循环时，选择哪一个维度，宗旨是复用尽可能大的数据块。
 *	   比如，在 Micro Kernel 的 i-l 内层循环中，
 *     本质上计算的是 C（MR * NR） += A（MR * 1） * B(1 * NR)，
 *	   其中 C 的 tile 尺寸最大，因此我们在外层套上 kr 的循环，从而复用寄存器中 C 的数据。
 *	   按照同样的思路，我们认为 KC 的选取大概率会使 B 的 tile 尺寸最大，
 *     因此在 Micro Kernel 的外层套上 ir 的循环。
 *     依此类推，循环的顺序是 jc-ic-pc-jr-ir-kr-i-l。
 *   - MR/NR 的选取应考虑ymm寄存器数量，尽可能地利用寄存器同时处理更多数据，
 *     并且高效地利用流水线。
 *   - KC 的选取应考虑 L1 缓存大小，以确保 A B C 的 tile 能够同时驻留在 L1 中，
 *     同时保证流水线的高效运行。通常 KC 要满足
 * 
 *			2 * MR * NR + 2 * MR * KC + KC * NR < L1 cache size / sizeof(float)
 *			   （C tile）    （A tile）（B tile）
 * 
 *	   其中预留两倍的 A 和 C tile 的空间，是为了循环时，
 *     后续的 A 和 C tile 加载时不会将需要复用的 B tile 驱逐出 L1。
 *     在这个前提下，KC 应尽量大，以充分利用 L1 的容量，减少对 L2 的访问。
 *   - MC 的选取应是 MR 的整数倍，并且要考虑 L2 缓存大小，
 *     以确保 A B C 的 tile 能够同时驻留在 L2 中，同时保证流水线的高效运行。
 *     通常 MC 要满足
 *
 *			2 * MC * NR + MC * KC + 2 * KC * NR < L2 cache size / sizeof(float)
 *			   （C tile）（A tile）    （B tile）
 *
 *	   其中预留两倍的 B 和 C tile 的空间，是为了循环时，
 *     后续的 B 和 C tile 加载时不会将需要复用的 A tile 驱逐出 L2。
 *     在这个前提下，MC 应尽量大，以充分利用 L2 的容量，减少对 L3 的访问。
 *   - NC 的选取应是 NR 的整数倍，并且要考虑 L3 缓存大小，
 *     以确保 A B C 的 tile 能够同时驻留在 L3 中，同时保证流水线的高效运行。
 *	   通常 NC 要满足
 *
 *			MC * NC + 2 * MC * KC + 2 * KC * NC < L3 cache size / sizeof(float)
 *		   （C tile）    （A tile）    （B tile）
 *
 *	   其中预留两倍的 A 和 B tile 的空间，是为了循环时，
 *     后续的 A 和 B tile 加载时不会将需要复用的 C tile 驱逐出 L3。
 *     在这个前提下，NC 应尽量大，以充分利用 L3 的容量，减少对内存的访问。
 */
void MatMul06(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int NC = GEMM_NC;
	constexpr int MC = GEMM_MC;
	constexpr int KC = GEMM_KC;
	constexpr int NR = GEMM_NR;
	constexpr int MR = GEMM_MR;
	constexpr int nr = NR >> 3;
	int currentNC, currentMC, currentKC;
	int currentNR, currentMR;
	float* macroA, * macroB, * macroC;
	float* microA, * microB, * microC;
	for (int jc = 0; jc < N; jc += NC)
	{
		currentNC = min(NC, N - jc);
		for (int ic = 0; ic < M; ic += MC)
		{
			currentMC = min(MC, M - ic);
			macroC = C + ic * ldc + jc;
			for (int pc = 0; pc < K; pc += KC)
			{
				currentKC = min(KC, K - pc);
				macroA = A + ic * lda + pc;
				macroB = B + pc * ldb + jc;
				//Macro Kernel
				for (int jr = 0; jr < currentNC; jr += NR)
				{
					currentNR = min(NR, currentNC - jr);
					microB = macroB + jr;
					if (currentNR % 8 != 0)
					{
						int l = currentNR & ~7;
						alignas(32) int mask_array[8]{ 0 };
						for (int ll = 0; ll < currentNR - l; ++ll)
							mask_array[ll] = -1;
						__m256i mask = _mm256_load_si256((const __m256i*)mask_array);

						for (int ir = 0; ir < currentMC; ir += MR)
						{
							currentMR = min(MR, currentMC - ir);
							microA = macroA + ir * lda;
							microC = macroC + ir * ldc + jr;
							//Micro Kernel
							__m256 c[MR][nr];
							__m256 b[nr];
							__m256 a;
							for (int i = 0; i < currentMR; i++)
							{
								for (l = 0; l <= currentNR - 8; l += 8)
									c[i][l >> 3] = _mm256_load_ps(microC + i * ldc + l);
								c[i][l >> 3] = _mm256_maskload_ps(microC + i * ldc + l, mask);
							}
							for (int kr = 0; kr < currentKC; kr++)
							{
								for (l = 0; l <= currentNR - 8; l += 8)
									b[l >> 3] = _mm256_load_ps(microB + kr * ldb + l);
								b[l >> 3] = _mm256_maskload_ps(microB + kr * ldb + l, mask);

								for (int i = 0; i < currentMR; i++)
								{
									a = _mm256_set1_ps(microA[i * lda + kr]);
									for (int l = 0; l < currentNR; l += 8)
										c[i][l >> 3] = _mm256_fmadd_ps(a, b[l >> 3], c[i][l >> 3]);
								}
							}
							for (int i = 0; i < currentMR; i++)
							{
								for (l = 0; l <= currentNR - 8; l += 8)
									_mm256_store_ps(microC + i * ldc + l, c[i][l >> 3]);
								_mm256_maskstore_ps(microC + i * ldc + l, mask, c[i][l >> 3]);
							}
						}
					}
					else
					{
						for (int ir = 0; ir < currentMC; ir += MR)
						{
							currentMR = min(MR, currentMC - ir);
							microA = macroA + ir * lda;
							microC = macroC + ir * ldc + jr;
							//Micro Kernel
							__m256 c[MR][nr];
							__m256 b[nr];
							__m256 a;
							for (int i = 0; i < currentMR; i++)
							{
								for (int l = 0; l < currentNR; l += 8)
									c[i][l >> 3] = _mm256_load_ps(microC + i * ldc + l);
							}
							for (int kr = 0; kr < currentKC; kr++)
							{
								for (int l = 0; l < currentNR; l += 8)
									b[l >> 3] = _mm256_load_ps(microB + kr * ldb + l);

								for (int i = 0; i < currentMR; i++)
								{
									a = _mm256_set1_ps(microA[i * lda + kr]);
									for (int l = 0; l < currentNR; l += 8)
										c[i][l >> 3] = _mm256_fmadd_ps(a, b[l >> 3], c[i][l >> 3]);
								}
							}
							for (int i = 0; i < currentMR; i++)
							{
								for (int l = 0; l < currentNR; l += 8)
									_mm256_store_ps(microC + i * ldc + l, c[i][l >> 3]);
							}
						}
					}
				}
			}
		}
	}
}

/**
 * @brief 在 MatMul06 基础上引入数据重排（Packing），进一步提升访存连续性和缓存利用。
 * 
 * 相比 MatMul06 的改进：
 *   - **显式数据打包**：MatMul06 直接从原始矩阵 A 和 B 中访问分块数据，
 *     导致在 Micro Kernel 内部访问 A 和 B 时存在跨行（lda/ldb）的非连续访存，
 *     且不同分块间可能相互驱逐缓存。
 *   - MatMul07 调用 PackA 和 PackB 将当前 KC×MC 和 KC×NC 的子块提前拷贝到
 *     连续缓冲区 APanel 和 BPanel 中，存储布局按 Micro Kernel 的需求优化：
 *       - APanel：按 MR 行块组织，每个行块内为列主序连续存储（方便按列加载 a 向量）。
 *       - BPanel：按 NR 列块组织，每个列块内为行主序连续存储（方便按行加载 b 向量）。
 *   - **简化 Micro Kernel**：由于打包后的数据在内存中完全连续且对齐，
 *     无需再处理非对齐尾部（假设 MR/NR 整除 MC/NC），
 *     也不再需要掩码加载/存储指令，代码更简洁高效。
 *   - **减少缓存缺失**：打包后的子块在 Micro Kernel 执行期间能够完全驻留在 L1/L2 缓存中，
 *     且对 A 和 B 的访问均为顺序流式访问，充分利用硬件预取。
 * 
 * @note 打包操作本身有额外开销，但对于大矩阵（K 较大），
 *       其带来的访存优化收益远超过拷贝成本。
 *       分块参数 MR/NR 通常与向量宽度（如 8）和寄存器数量相匹配。
 */
void MatMul07(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int NC = GEMM_NC;
	constexpr int MC = GEMM_MC;
	constexpr int KC = GEMM_KC;
	constexpr int NR = GEMM_NR;
	constexpr int MR = GEMM_MR;
	constexpr int nr = NR >> 3;
	int currentNC, currentMC, currentKC;
	float* macroA, * macroB, * macroC;
	float* microA, * microB, * microC;
	float* APanel = (float*)_aligned_malloc(sizeof(float) * MC * KC, 32);
	float* BPanel = (float*)_aligned_malloc(sizeof(float) * KC * NC, 32);
	for (int jc = 0; jc < N; jc += NC)
	{
		currentNC = min(NC, N - jc);
		for (int ic = 0; ic < M; ic += MC)
		{
			currentMC = min(MC, M - ic);
			macroC = C + ic * ldc + jc;
			for (int pc = 0; pc < K; pc += KC)
			{
				currentKC = min(KC, K - pc);
				macroA = A + ic * lda + pc;
				macroB = B + pc * ldb + jc;
				PackA(lda, currentMC, currentKC, MR, macroA, APanel);
				PackB(ldb, currentKC, currentNC, NR, macroB, BPanel);
				//Macro Kernel
				for (int jr = 0; jr < currentNC; jr += NR)
				{
					microB = BPanel + jr * currentKC;
					for (int ir = 0; ir < currentMC; ir += MR)
					{
						microA = APanel + ir * currentKC;
						microC = macroC + ir * ldc + jr;
						//Micro Kernel
						__m256 c[MR][nr];
						__m256 b[nr];
						__m256 a;
						for (int i = 0; i < MR; i++)
							for (int l = 0; l < nr; l++)
								c[i][l] = _mm256_load_ps(microC + i * ldc + l * 8);
						for (int kr = 0; kr < currentKC; kr++)
						{
							for (int l = 0; l < nr; l++)
								b[l] = _mm256_load_ps(microB + kr * NR + l * 8);

							for (int i = 0; i < MR; i++)
							{
								a = _mm256_set1_ps(microA[kr * MR + i]);
								for (int l = 0; l < nr; l++)
									c[i][l] = _mm256_fmadd_ps(a, b[l], c[i][l]);
							}
						}
						for (int i = 0; i < MR; i++)
							for (int l = 0; l < nr; l++)
								_mm256_store_ps(microC + i * ldc + l * 8, c[i][l]);
					}
				}
			}
		}
	}
	_aligned_free(APanel);
	_aligned_free(BPanel);
}

/**
 * @brief 在 MatMul07 基础上调整循环顺序，减少 B 矩阵的重复打包开销。
 *
 * 相比 MatMul07 的改进：
 *   - MatMul07 的循环顺序为 jc → ic → pc，导致对于每个 ic 行块，都会重新打包一次 B 的当前列块（pc 内层循环），
 *     即 BPanel 会被重复打包多次（次数 = ic 的迭代次数），造成大量冗余拷贝。
 *   - MatMul08 将循环顺序调整为 jc → pc → ic，先固定 pc（K 维度块）和 jc（N 维度块），
 *     打包一次 B 得到 BPanel，然后在内层 ic 循环中复用该 BPanel 处理所有 MC 行块。
 *   - 这样 B 的打包次数从 O(M/MC * K/KC) 降为 O(K/KC)，大幅减少数据重排开销，
 *     特别适合 M 较大、K 较大的场景。
 *
 * @note 循环顺序的调整不改变计算结果，但优化了内存访问模式和数据复用。
 */
void MatMul08(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int NC = GEMM_NC;
	constexpr int MC = GEMM_MC;
	constexpr int KC = GEMM_KC;
	constexpr int NR = GEMM_NR;
	constexpr int MR = GEMM_MR;
	constexpr int nr = NR >> 3;
	int currentNC, currentMC, currentKC;
	float* macroA, * macroB, * macroC;
	float* microA, * microB, * microC;
	float* APanel = (float*)_aligned_malloc(sizeof(float) * MC * KC, 32);
	float* BPanel = (float*)_aligned_malloc(sizeof(float) * KC * NC, 32);
	for (int jc = 0; jc < N; jc += NC)
	{
		currentNC = min(NC, N - jc);
		for (int pc = 0; pc < K; pc += KC)
		{
			currentKC = min(KC, K - pc);
			macroB = B + pc * ldb + jc;
			PackB(ldb, currentKC, currentNC, NR, macroB, BPanel);
			for (int ic = 0; ic < M; ic += MC)
			{
				currentMC = min(MC, M - ic);
				macroA = A + ic * lda + pc;
				macroC = C + ic * ldc + jc;
				PackA(lda, currentMC, currentKC, MR, macroA, APanel);
				//Macro Kernel
				for (int jr = 0; jr < currentNC; jr += NR)
				{
					microB = BPanel + jr * currentKC;
					for (int ir = 0; ir < currentMC; ir += MR)
					{
						microA = APanel + ir * currentKC;
						microC = macroC + ir * ldc + jr;
						//Micro Kernel
						__m256 c[MR][nr];
						__m256 b[nr];
						__m256 a;
						for (int i = 0; i < MR; i++)
							for (int l = 0; l < nr; l++)
								c[i][l] = _mm256_load_ps(microC + i * ldc + l * 8);
						for (int kr = 0; kr < currentKC; kr++)
						{
							for (int l = 0; l < nr; l++)
								b[l] = _mm256_load_ps(microB + kr * NR + l * 8);

							for (int i = 0; i < MR; i++)
							{
								a = _mm256_set1_ps(microA[kr * MR + i]);
								for (int l = 0; l < nr; l++)
									c[i][l] = _mm256_fmadd_ps(a, b[l], c[i][l]);
							}
						}
						for (int i = 0; i < MR; i++)
							for (int l = 0; l < nr; l++)
								_mm256_store_ps(microC + i * ldc + l * 8, c[i][l]);
					}
				}
			}
		}
	}
	_aligned_free(APanel);
	_aligned_free(BPanel);
}

/**
 * @brief 在 MatMul08 基础上引入 OpenMP 多线程并行，充分利用多核 CPU。
 *
 * 相比 MatMul08 的改进：
 *   - MatMul08 为单线程执行，对于大规模矩阵无法利用多核资源。
 *   - MatMul09 使用 OpenMP 并行化 M 维度的外循环（ic 循环），将整个 M 方向按 MR 块均匀分配到多个线程。
 *   - 每个线程拥有独立的 APanel 缓冲区（APanel + tid * MC * currentKC），避免多线程同时写入时的数据竞争。
 *   - 通过计算每个线程负责的行块范围（start/end），实现负载均衡（考虑块数不能整除时的余数分配）。
 *   - B 矩阵的打包（BPanel）仍为单线程完成，且在所有线程间共享（只读），减少了重复打包开销。
 *
 * @note 线程数由宏 OMP_THREADS 控制，需根据 CPU 核心数合理设置。
 *       该并行策略对 M 较大的情况加速效果显著。
 */
void MatMul09(const int M, const int N, const int K, const int lda, const int ldb, const int ldc, float* A, float* B, float* C)
{
	constexpr int NC = GEMM_NC;
	constexpr int MC = GEMM_MC;
	constexpr int KC = GEMM_KC;
	constexpr int NR = GEMM_NR;
	constexpr int MR = GEMM_MR;
	constexpr int nr = NR >> 3;
	const int numBlocks = (M - 1 + MR) / MR;
	int currentNC, currentKC;
	float* macroB;
	float* APanel = (float*)_aligned_malloc(sizeof(float) * MC * KC * OMP_THREADS, 32);
	float* BPanel = (float*)_aligned_malloc(sizeof(float) * KC * NC, 32);
	for (int jc = 0; jc < N; jc += NC)
	{
		currentNC = min(NC, N - jc);
		for (int pc = 0; pc < K; pc += KC)
		{
			currentKC = min(KC, K - pc);
			macroB = B + pc * ldb + jc;
			PackB(ldb, currentKC, currentNC, NR, macroB, BPanel);

			#pragma omp parallel num_threads(OMP_THREADS)
			{
				const int numThreads = omp_get_num_threads();
				const int tid = omp_get_thread_num();
				float* currentAPanel = APanel + tid * MC * currentKC;

				const int numBlocksPerThread = numBlocks / numThreads;
				const int numBlocksLeft = numBlocks % numThreads;
				const int start = (tid * numBlocksPerThread + min(tid, numBlocksLeft)) * MR;
				const int end = min(((tid + 1) * numBlocksPerThread + min((tid + 1), numBlocksLeft)) * MR, M);

				int currentMC;
				float* macroA, * macroC;
				float* microA, * microB, * microC;
				for (int ic = start; ic < end; ic += MC)
				{
					currentMC = min(MC, end - ic);
					macroA = A + ic * lda + pc;
					macroC = C + ic * ldc + jc;
					PackA(lda, currentMC, currentKC, MR, macroA, currentAPanel);
					//Macro Kernel
					for (int jr = 0; jr < currentNC; jr += NR)
					{
						microB = BPanel + jr * currentKC;
						for (int ir = 0; ir < currentMC; ir += MR)
						{
							microA = currentAPanel + ir * currentKC;
							microC = macroC + ir * ldc + jr;
							//Micro Kernel
							__m256 c[MR][nr];
							__m256 b[nr];
							__m256 a;
							for (int i = 0; i < MR; i++)
								for (int l = 0; l < nr; l++)
									c[i][l] = _mm256_load_ps(microC + i * ldc + l * 8);
							for (int kr = 0; kr < currentKC; kr++)
							{
								for (int l = 0; l < nr; l++)
									b[l] = _mm256_load_ps(microB + kr * NR + l * 8);
								for (int i = 0; i < MR; i++)
								{
									a = _mm256_set1_ps(microA[kr * MR + i]);
									for (int l = 0; l < nr; l++)
										c[i][l] = _mm256_fmadd_ps(a, b[l], c[i][l]);
								}
							}
							for (int i = 0; i < MR; i++)
								for (int l = 0; l < nr; l++)
									_mm256_store_ps(microC + i * ldc + l * 8, c[i][l]);
						}
					}
				}
			}
		}
	}
	_aligned_free(APanel);
	_aligned_free(BPanel);
}