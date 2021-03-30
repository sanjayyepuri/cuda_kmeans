#include <cuda_runtime.h>
#include <math.h>

#include "kmeans.h"

namespace kmeans
{
    namespace kcuda
    {
        static __device__ float distance(float *a, float *b, size_t dims)
        {
            float dist = 0;
            for (int i = 0; i < dims; ++i)
            {
                float diff = a[i] - b[i];
                dist += diff * diff;
            }

            return dist;
        }

        // this is not fast :P
        __global__ void computeMinDistanceKern(float *dist, float *vecs, float *centroids,
                                               size_t N, size_t K, size_t D)
        {
            int x_ind = blockIdx.x * blockDim.x + threadIdx.x;
            float *X = &vecs[I(x_ind, 0, D)];

            float min_dist = MAXFLOAT;
            for (int c = 0; c < K; ++c) 
                min_dist = fmin(min_dist, distance(X, &centroids[I(c, 0, D)], D));

            dist[x_ind] = min_dist;
        }
    }

    void computeMinDistance(float *dist, float *vecs, float *centroids, size_t N, size_t K, size_t D)
    {
        size_t ds_size = sizeof(float) * N * D; 
        size_t centroid_size = sizeof(float) * K * D;
        size_t dist_size = sizeof(float) * N;

        // allocate buffers on device
        float *cuda_vecs, *cuda_centroids, *cuda_dist;
        CUDA_ERR(cudaMalloc(&cuda_vecs, ds_size));
        CUDA_ERR(cudaMalloc(&cuda_centroids, centroid_size));
        CUDA_ERR(cudaMalloc(&cuda_dist, dist_size));

        // load data
        CUDA_ERR(cudaMemcpy(cuda_vecs, vecs, ds_size, cudaMemcpyHostToDevice));
        CUDA_ERR(cudaMemcpy(cuda_centroids, centroids, centroid_size, cudaMemcpyHostToDevice));

        // TODO a better way to do this 
        const size_t N_THREADS = 1024;
        dim3 threads(N_THREADS);
        dim3 blocks((N + N_THREADS - 1) / N_THREADS);

        kcuda::computeMinDistanceKern <<<blocks, threads>>>(cuda_dist, cuda_vecs, cuda_centroids, N, K, D);
        CUDA_ERR(cudaPeekAtLastError());
        CUDA_ERR(cudaDeviceSynchronize());

        CUDA_ERR(cudaMemcpy(dist, cuda_dist, dist_size, cudaMemcpyDeviceToHost));
    }

    void kmeansppInitCentroids(Dataset &ds)
    {
        int first_centroid = (int)(rand_float() * ds.n);
        ds.init_centroids.push_back(first_centroid);

        float *centroids = new float[ds.dims * ds.k];
        float *D = new float[ds.n];

        while (ds.init_centroids.size() < ds.k)
        {
            copyCentroids(ds, centroids); // performing unecessary copies
            computeMinDistance(D, ds.vecs, centroids, ds.n, ds.init_centroids.size(), ds.dims);

            float total_dist = 0.0f;
            for (int i = 0; i < ds.n; ++i)
                total_dist += D[i]; // we are computing distance square

            float target = rand_float() * total_dist;
            float dist = 0.0f;
            for (int i = 0; i < ds.n; ++i)
            {
                dist += D[i];
                if (target < dist) {
                    ds.init_centroids.push_back(i);
                    break;
                }
            }
        }
    }

}