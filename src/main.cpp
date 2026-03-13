#include <iostream>
#include <sstream>
#include <utils.h>
#include <matmul.h>
#include <omp.h>

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
		default:
			break;
	}

	const float tolerance = 1e-5;
	const int nrepeats = 4;

	float* A, * B, * C, * GT;
	MallocMatrix(M, N, K, A, B, C, GT);

	InitABCGT(M, N, K, A, B, C, GT);

	for (size_t i = 0; i < nrepeats; i++)
		MatMulGT(M, N, K, A, B, GT);

	for (size_t i = 0; i < nrepeats; i++)
		f(M, N, K, A, B, C);

	CheckResult(M, N, C, GT, tolerance);

	FreeMatrix(A, B, C, GT);
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