#include <iostream>
#include <sstream>
#include <utils.h>

using std::cout;
using std::endl;
using std::istringstream;
using std::cerr;

#define A(i, k) A[(i) * K + (k)] // M * K
#define B(k, j) B[(k) * N + (j)] // K * N
#define C(i, j) C[(i) * N + (j)] // M * N

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

	float* A = (float*)malloc(sizeof(float) * M * K);
	float* B = (float*)malloc(sizeof(float) * K * N);
	float* C = (float*)malloc(sizeof(float) * M * N);

	InitABC(M, N, K, A, B, C);

	free(A);
	free(B);
	free(C);
}