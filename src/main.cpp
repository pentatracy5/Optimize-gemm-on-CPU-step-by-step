#include <iostream>
#include <sstream>
#include <chrono>
#include <utils.h>
#include <matmul.h>
#include <config.h>

using std::cout;
using std::endl;
using std::istringstream;
using std::cerr;

void Test(int M, int N, int K, unsigned int version)
{
	constexpr int totalVersions = sizeof(matmulFuncs) / sizeof(matmulFuncs[0]);
	if (version >= totalVersions) version = totalVersions - 1;

	MatMulFunc f{ matmulFuncs[version] };
	MatMulFunc ref{ MatMulREF };

	constexpr float tolerance = TOLERANCE;
	constexpr int nrepeats = NREPEATS;
	constexpr int warmup = WARMUP;

	float* A, * B, * C, * REF;
	int lda, ldb, ldc;
	MallocMatrix(M, N, K, lda, ldb, ldc, A, B, C, REF);

	InitABCREF(M, N, K, lda, ldb, ldc, A, B, C, REF);

	for (int i = 0; i < warmup; ++i) {
		ref(M, N, K, lda, ldb, ldc, A, B, REF);
		f(M, N, K, lda, ldb, ldc, A, B, C);
	}

	std::chrono::duration<double> elapsed(0);
	for (int i = 0; i < nrepeats; i++)
	{
		ClearCache();
		auto start = std::chrono::high_resolution_clock::now();
		ref(M, N, K, lda, ldb, ldc, A, B, REF);
		auto end = std::chrono::high_resolution_clock::now();
		elapsed += end - start;
	}
	double time_ref = elapsed.count() / nrepeats;

	elapsed = std::chrono::duration<double>::zero();
	for (int i = 0; i < nrepeats; i++)
	{
		ClearCache();
		auto start = std::chrono::high_resolution_clock::now();
		f(M, N, K, lda, ldb, ldc, A, B, C);
		auto end = std::chrono::high_resolution_clock::now();
		elapsed += end - start;
	}
	double time_f = elapsed.count() / nrepeats;

	double flops = 2 * M / 1000.0 * N / 1000.0 * K / 1000.0;
	cout << "M\tN\tK\tref_GFLOPS\tf_GFLOPS" << endl;
	cout << M << '\t' << N << '\t' << K << '\t' << flops / time_ref << '\t' << flops / time_f << endl;

	CheckResult(M, N, ldc, C, REF, tolerance);

	FreeMatrix(A, B, C, REF);
}

void Run(int M, int N, int K, unsigned int version)
{
	constexpr int totalVersions = sizeof(matmulFuncs) / sizeof(matmulFuncs[0]);
	if (version >= totalVersions) version = totalVersions - 1;

	MatMulFunc f{ matmulFuncs[version] };

	constexpr int nrepeats = NREPEATS;

	float* A, * B, * C, * REF;
	int lda, ldb, ldc;
	MallocMatrix(M, N, K, lda, ldb, ldc, A, B, C, REF);

	InitABCREF(M, N, K, lda, ldb, ldc, A, B, C, REF);

	for (int i = 0; i < nrepeats; ++i) {
		f(M, N, K, lda, ldb, ldc, A, B, C);
	}

	FreeMatrix(A, B, C, REF);
}

int main(int argc, char* argv[])
{
	int M, N, K;
	unsigned int version;
	if (argc != 5)
	{
		cout << "Error: require 4 arguments, but " << argc - 1 << " provided." << endl;
		return 1;
	}

	istringstream iss1(argv[1]), iss2(argv[2]), iss3(argv[3]), iss4(argv[4]);
	if (!(iss1 >> M) || !(iss2 >> N) || !(iss3 >> K))
	{
		cerr << "Error: invalid integer arguments." << endl;
		return 1;
	}
	if (!(iss4 >> version)) {
		cerr << "Error: invalid matmul version." << endl;
		return 1;
	}
	if (M <= 0 || N <= 0 || K <= 0)
	{
		cerr << "Error: invalid arguments integer value." << endl;
		return 1;
	}

	if constexpr (PROFILE)
		Run(M, N, K, version);
	else
		Test(M, N, K, version);

	return 0;
}