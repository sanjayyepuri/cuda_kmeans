#include <cuda_runtime.h>

#include "kmeans.h"

namespace kmeans
{

    __global__ void convergenceKernel(float *o_centroids, float *n_centroiods, float threshold, int )

    Labels kmeansGPU(const Dataset &ds, const Args &options)
    {
        size_t N = ds.n, D = ds.dims, K = ds.k;

        size_t ds_size = sizeof(float) * N * D;
        size_t centroid_size = sizeof(float) * N * K;

        float *centroids[2];
        centroids[0] = new float[K * D];
        centroids[1] = new float[K * D];

        copyCentroids(ds, centroids[0]);

        float *cuda_dataset, *cuda_centroids[2];

        cudaMalloc(&cuda_dataset, sizeof(float) * N *  D);
        cudaMalloc(&cuda_centroids[0], sizeof(float) * K * D);
        cudaMalloc(&cuda_centroids[1], sizeof(float) * K * D);

        // copy dataset to host
        cudaMemcpy(cuda_dataset, ds.vecs, sizeof(float) * N * D, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_centroids, centroids[0], sizeof(float) * K * D, cudaMemcpyHostToDevice)
    }
}