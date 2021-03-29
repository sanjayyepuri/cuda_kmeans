#include <cuda_runtime.h>
#include <math.h>

#include "kmeans.h"

namespace kmeans
{
   namespace kcuda { 
        __device__
        float distance(float *a, float *b, size_t dims)
        {
            float dist = 0;
            for (int i = 0; i < dims; ++i)
            {
                float diff = a[i] - b[i];
                dist += diff * diff;
            }

            return dist;
        }

        __device__
        void add(float *a, float *b, size_t dims){
            for (int i = 0; i < dims; ++i)
                atomicAdd(&a[i], b[i]);
        }

        __global__ 
        void labelVectors(int iter, 
            float *vecs, float *centroids[2], 
            float *counts, int *labels,
            size_t N, size_t K, size_t D)
        {
            int x_ind = blockIdx.x * blockDim.x + threadIdx.x;

            float *X = &vecs[I(x_ind, 0, D)];
            
            int cc = (iter - 1) % 2; 
            int nc = iter % 2; 

            float min_dist = MAXFLOAT;
            int centroid_id = -1;

            for (int c = 0; c < K; ++c)
            {
                float *centroid = &centroids[cc][I(c, 0, D)];
                float dist = distance(X, centroid, D);

                if (dist < min_dist)
                {
                    min_dist = dist; 
                    centroid_id = c;
                }
            }

            labels[x_ind] = centroid_id;

            add(&centroids[nc][I(centroid_id, 0, D)], X, D);
            atomicAdd(&counts[centroid_id], 1.0f);
        }

        __global__
        void averageCentroids(float *centroids, float *counts, int D)
        {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            centroids[ind] /= counts[ind / D];
        }
    }

    Labels kmeansGPU(const Dataset &ds, const Args &options)
    {
        size_t N = ds.n, D = ds.dims, K = ds.k;

        size_t ds_size = sizeof(float) * N * D;
        size_t centroid_size = sizeof(float) * K * D;

        Labels l; 

        // copy centroids from dataset 
        l.centroids = new float[K * D];
        copyCentroids(ds, l.centroids);

        // double buffer centroids
        float *cuda_dataset, **cuda_centroids, *cuda_counts, *centroids[2];
        int *cuda_labels;


        CUDA_ERR(cudaMalloc(&cuda_dataset, ds_size));
        CUDA_ERR(cudaMalloc(&centroids[0], centroid_size));
        CUDA_ERR(cudaMalloc(&centroids[1], centroid_size));
        CUDA_ERR(cudaMalloc(&cuda_centroids, sizeof(float *) * 2));
        CUDA_ERR(cudaMalloc(&cuda_counts, sizeof(float) * K));
        CUDA_ERR(cudaMalloc(&cuda_labels, sizeof(int) * N));
        
        // copy dataset to device
        CUDA_ERR(cudaMemcpy(cuda_dataset, ds.vecs, ds_size, 
            cudaMemcpyHostToDevice));
        CUDA_ERR(cudaMemcpy(centroids[0], l.centroids, centroid_size,
            cudaMemcpyHostToDevice));
        CUDA_ERR(cudaMemcpy(cuda_centroids, centroids, sizeof(float *) * 2,
            cudaMemcpyHostToDevice));

        const size_t N_THREADS = 1024;

        std::vector<double> ms_per_iter;
        int iter = 0;
        while (iter++ < options.max_iters)
        {
            TIME_EXEC(ms_per_iter, {
                // zero second buffer and counts
                CUDA_ERR(cudaMemset(centroids[iter % 2], 0, centroid_size));
                CUDA_ERR(cudaMemset(cuda_counts, 0, sizeof(float) * K));
                CUDA_ERR(cudaDeviceSynchronize());

                kcuda::labelVectors <<< (N + N_THREADS - 1) / N_THREADS, N_THREADS >>>  (
                    iter,
                    cuda_dataset, cuda_centroids, 
                    cuda_counts, cuda_labels,
                    N, K, D
                );
                CUDA_ERR(cudaPeekAtLastError());
                CUDA_ERR(cudaDeviceSynchronize());

                kcuda::averageCentroids <<< 1, K * D >>> (centroids[iter % 2], cuda_counts, D);
                CUDA_ERR(cudaPeekAtLastError());
                CUDA_ERR(cudaDeviceSynchronize());
            })
        }

        printTimeMs(ms_per_iter);

        // copy results back to host 
        l.labels = new int[N];
        CUDA_ERR(cudaMemcpy(l.centroids, centroids[0], centroid_size,
            cudaMemcpyDeviceToHost));
        CUDA_ERR(cudaMemcpy(l.labels, cuda_labels, sizeof(int) * N,
            cudaMemcpyDeviceToHost));

        return l;
    }
}