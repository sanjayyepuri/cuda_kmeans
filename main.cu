#include <iostream>
#include <string>
#include <cuda_runtime.h>

__global__ void check_index()
{
    printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) "
    "blockDim (%d, %d, %d) gridDim(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, 
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.z, gridDim.z);
}

__global__ void sum_arr_dev(float *a, float *b, float *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void sum_arr_host(float *a, float *b, float *c, const int dims) 
{
    for (int i = 0; i < dims; ++i) 
    {
        c[i] = a[i] + b[i];
    }
}

void init_vec(float *vec, const int dims) 
{
    time_t t;
    srand((unsigned int) time(&t)); 

    for (int i = 0; i < dims; ++i) 
    {
        vec[i] = (float) (rand() & 0xFF) / 10.0f; 
    }
}

void print_vec(float *vec, const int dims) 
{
    std::cout << "[" << vec[0];
    for (int i = 1; i < dims; ++i) 
        std::cout << ", " << vec[i]; 
    std::cout << "]" << std::endl;
}

float diff(float *A, float *B, const int dims) 
{
    float d = 0;
    for (int i = 0; i < dims; ++i)
        d += A[i] - B[i];

    return d;
}

int main() 
{
    size_t N = 1024; 
    size_t vec_size = sizeof(float) * N;

    dim3 block (1);
    dim3 grid ((N + block.x - 1) / block.x);

    float *A, *B, *C, *dev_result; 

    A = (float *)malloc(vec_size); // new float[N];
    B = (float *)malloc(vec_size);
    C = (float *)malloc(vec_size);
    dev_result = (float *)malloc(vec_size);

    init_vec(A, N); 
    init_vec(B, N);

    float *A_dev, *B_dev, *C_dev; 

    cudaMalloc(&A_dev, vec_size);
    cudaMalloc(&B_dev, vec_size);
    cudaMalloc(&C_dev, vec_size);

    cudaMemcpy(A_dev, A, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, vec_size, cudaMemcpyHostToDevice);

    sum_arr_dev <<<block, grid>>> (A_dev, B_dev, C_dev);
    cudaMemcpy(dev_result, C_dev, vec_size, cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev); 

    sum_arr_host(A, B, C, N); 

    std::cout << diff(C, dev_result, N) << std::endl;

    free(A); // delete[] A;
    free(B);
    free(C); 

    return 0;
}
