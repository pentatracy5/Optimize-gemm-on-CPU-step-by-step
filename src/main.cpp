#include <iostream>
#include <sstream>
#include <utils.h>
#include <matmul.h>
#include <chrono>

using std::cout;
using std::endl;
using std::istringstream;
using std::cerr;

void Test(int M, int N, int K, int version)
{
	MatMulFunc f{ MatMul0 };
	switch (version)
	{
		case 0:
			f = MatMul0;
			break;
		case 1:
			f = MatMul1;
			break;
		case 2:
			f = MatMul2;
			break;
		case 3:
			f = MatMul3;
			break;
		default:
			break;
	}

	const float tolerance = 1e-1;
	const int nrepeats = 10;
	const int warmup = 2;

	float* A, * B, * C, * REF;
	MallocMatrix(M, N, K, A, B, C, REF);

	InitABCREF(M, N, K, A, B, C, REF);

	for (int i = 0; i < warmup; ++i) {
		MatMulREF(M, N, K, A, B, REF);
		f(M, N, K, A, B, C);
	}

	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < nrepeats; i++)
		MatMulREF(M, N, K, A, B, REF);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	double time_ref = elapsed.count() / nrepeats;

	start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < nrepeats; i++)
		f(M, N, K, A, B, C);
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	double time_f = elapsed.count() / nrepeats;

	double flops = 2 * M / 1000.0 * N / 1000.0 * K / 1000.0;
	cout << "M\tN\tK\tref_GFLOPS\tf_GFLOPS" << endl;
	cout << M << '\t' << N << '\t' << K << '\t' << flops / time_ref << '\t' << flops / time_f << endl;

	CheckResult(M, N, C, REF, tolerance);

	FreeMatrix(A, B, C, REF);
}

int main(int argc, char* argv[])
{
	int M, N, K, version;
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

	Test(M, N, K, version);
}