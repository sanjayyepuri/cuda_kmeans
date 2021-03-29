#pragma once

#include <vector>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

#define BSTR(b) (b ? "true" : "false")
#define DEBUG

#define I(r, c, D) (D*r + c)

#define TIME_EXEC(vec, ...) do {                                                \
    auto __startt = std::chrono::steady_clock::now();                           \
    __VA_ARGS__                                                                 \
    auto __endt = std::chrono::steady_clock::now();                             \
    std::chrono::duration<double> __duration = __endt - __startt;               \
    vec.push_back(__duration.count() * 1000);                                   \
} while(0);


#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace kmeans
{
    struct Args {
        int k = 0;
        int dims = 0;
        int max_iters = 20;
        int rand_seed = 8675309;

        char *input_file;
        float threshold = 1e-10;

        bool print_centroids = false;
        bool gpu = false;
        bool gpu_shmem = false;
        bool kmeans_pp = false;
    };

    // store the dataset used for kmeans
    struct Dataset  {
        float *vecs;
        size_t n;
        size_t k;
        size_t dims;
        std::vector<size_t> init_centroids;
    };

    struct Labels {
        float *centroids;
        int *labels;
    };

    // return a dataset initalized with random centroids
    Dataset buildDataset(Args &options);

    // Sequential kmeans implementation
    Labels kmeansSequential(const Dataset &ds, const Args &options);

    // GPU kmeans implementation
    Labels kmeansGPU(const Dataset &ds, const Args &options);

    // compute average of vector of floats
    void printTimeMs(std::vector<double> ms_per_iter);

    // copy the initial centroids from the dataset
    void copyCentroids(const Dataset &ds, float *centroids);
}


