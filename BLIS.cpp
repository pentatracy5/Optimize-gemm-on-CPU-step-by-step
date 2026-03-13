#include <iostream>

using std::cout;
using std::endl;

#define A(i, k) A[(i) * K + (k)] // M * K
#define B(k, j) B[(k) * N + (j)] // K * N
#define C(i, j) C[(i) * N + (j)] // M * N

int main(int argc, char* argv[])
{
	int M;
	int N;
	int K;
	if (argc != 4)
	{
		cout << "Error: require 3 arguments, but " << argc - 1 << " provided." << endl;
		exit(0);
	}
}