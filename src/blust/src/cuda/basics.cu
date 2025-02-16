#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; ++i)
        y[i] += x[i];
}

int main()
{
    int N = 1 << 20;

    float *x, *y;

    // x = new float[N];
    // y = new float[N];
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i)
        maxErr = fmax(maxErr, fabs(y[i] - 3.0f));
    std::cout << "Max err: " << maxErr << '\n';

    cudaFree(x);
    cudaFree(y);

    return 0;
}
