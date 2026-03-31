#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <utils.h>
#include <matmul.h>
#include <config.h>

using std::cout;
using std::endl;
using std::istringstream;
using std::cerr;
using std::vector;

vector<int> v;
void ClearCache()
{
	constexpr size_t cache_size = 32 * 1024 * 1024;
	constexpr size_t num_elements = cache_size / sizeof(int);
	v.resize(num_elements, 0);
	for (size_t i = 0; i < num_elements; i++)
		v[i] = i;
}

void Test(size_t M, size_t N, size_t K, unsigned int version)
{
	constexpr size_t totalVersions = sizeof(matmulFuncs) / sizeof(matmulFuncs[0]);
	if (version >= totalVersions) version = totalVersions - 1;

	MatMulFunc f{ matmulFuncs[version] };
	MatMulFunc ref{ MatMulREF };

	constexpr float tolerance = TOLERANCE;
	constexpr size_t nrepeats = NREPEATS;
	constexpr size_t warmup = WARMUP;

	float* A, * B, * C, * REF;
	MallocMatrix(M, N, K, A, B, C, REF);

	InitABCREF(M, N, K, A, B, C, REF);

	for (size_t i = 0; i < warmup; ++i) {
		ref(M, N, K, A, B, REF);
		f(M, N, K, A, B, C);
	}

	std::chrono::duration<double> elapsed(0);
	for (size_t i = 0; i < nrepeats; i++)
	{
		ClearCache();
		auto start = std::chrono::high_resolution_clock::now();
		ref(M, N, K, A, B, REF);
		auto end = std::chrono::high_resolution_clock::now();
		elapsed += end - start;
	}
	double time_ref = elapsed.count() / nrepeats;

	elapsed = std::chrono::duration<double>::zero();
	for (size_t i = 0; i < nrepeats; i++)
	{
		ClearCache();
		auto start = std::chrono::high_resolution_clock::now();
		f(M, N, K, A, B, C);
		auto end = std::chrono::high_resolution_clock::now();
		elapsed += end - start;
	}
	double time_f = elapsed.count() / nrepeats;

	double flops = 2 * M / 1000.0 * N / 1000.0 * K / 1000.0;
	cout << "M\tN\tK\tref_GFLOPS\tf_GFLOPS" << endl;
	cout << M << '\t' << N << '\t' << K << '\t' << flops / time_ref << '\t' << flops / time_f << endl;

	CheckResult(M, N, C, REF, tolerance);

	FreeMatrix(A, B, C, REF);
}

void Run(size_t M, size_t N, size_t K, unsigned int version)
{
	constexpr size_t totalVersions = sizeof(matmulFuncs) / sizeof(matmulFuncs[0]);
	if (version >= totalVersions) version = totalVersions - 1;

	MatMulFunc f{ matmulFuncs[version] };

	constexpr size_t nrepeats = NREPEATS;

	float* A, * B, * C, * REF;
	MallocMatrix(M, N, K, A, B, C, REF);

	InitABCREF(M, N, K, A, B, C, REF);

	for (size_t i = 0; i < nrepeats; ++i) {
		f(M, N, K, A, B, C);
	}

	FreeMatrix(A, B, C, REF);
}

int main(int argc, char* argv[])
{
	size_t M, N, K;
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

#ifdef PROFILE
	Run(M, N, K, version);
#else
	Test(M, N, K, version);
#endif
	return 0;
}