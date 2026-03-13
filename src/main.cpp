#include <iostream>
#include <sstream>
#include <utils.h>
#include <matmul.h>

using std::cout;
using std::endl;
using std::istringstream;
using std::cerr;

int main(int argc, char* argv[])
{
	int M, N, K;
	if (argc != 4)
	{
		cout << "Error: require 3 arguments, but " << argc - 1 << " provided." << endl;
		return 1;
	}

	istringstream iss1(argv[1]), iss2(argv[2]), iss3(argv[3]);
	if (!(iss1 >> M) || !(iss2 >> N) || !(iss3 >> K)) {
		cerr << "Error: invalid integer arguments." << endl;
		return 1;
	}
	if (M <= 0 || N <= 0 || K <= 0)
	{
		cerr << "Error: invalid arguments integer value." << endl;
		return 1;
	}

	float *A, *B, *C, *GT;
	MallocMatrix(M, N, K, A, B, C, GT);

	InitAB(M, N, K, A, B);
	InitC(M, N, C);
	InitC(M, N, GT);
	MatMul0(M, N, K, A, B, GT);
	
	MatMul0(M, N, K, A, B, C);

	CheckResult(M, N, C, GT);
	
	PrintABC(M, N, K, A, B, C);

	FreeMatrix(A, B, C, GT);
}