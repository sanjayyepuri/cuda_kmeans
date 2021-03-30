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

        __device__ void add(float *a, float *b, size_t dims)
        {
            for (int i = 0; i < dims; ++i)
                atomicAdd(&a[i], b[i]);
        }

        __global__ void labelVectors(int iter,
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

        __global__ void labelVectorsShared(int iter,
                                           float *vecs, float *centroids[2],
                                           float *counts, int *labels,
                                           size_t N, size_t K, size_t D)
        {
            extern __shared__ float s[];
            float *local_centroids = s;
            float *centroid_scratch = &s[K * D];
            float *counts_scratch = &centroid_scratch[K * D];

            int bid = blockIdx.x, tid = threadIdx.x;
            int cc = (iter - 1) % 2;
            int nc = iter % 2;

            for (int i = tid; i < K * D; i += blockDim.x)
            {
                local_centroids[i] = centroids[cc][i];
                centroid_scratch[i] = 0;
            }

            for (int i = tid; i < K; i += blockDim.x)
                counts_scratch[i] = 0;

            __syncthreads();

            int x_ind = bid * blockDim.x + tid;
            float *X = &vecs[I(x_ind, 0, D)];

            float min_dist = MAXFLOAT;
            int centroid_id = -1;

            for (int c = 0; c < K; ++c)
            {
                float *centroid = &local_centroids[I(c, 0, D)];
                float dist = distance(X, centroid, D);

                if (dist < min_dist)
                {
                    min_dist = dist;
                    centroid_id = c;
                }
            }
            add(&centroid_scratch[I(centroid_id, 0, D)], X, D);
            atomicAdd(&counts_scratch[centroid_id], 1.0f);

            __syncthreads();

            labels[x_ind] = centroid_id;

            for (int i = tid; i < K * D; i += blockDim.x)
                atomicAdd(&centroids[nc][i], centroid_scratch[i]);

            for (int i = tid; i < K; i += blockDim.x)
                atomicAdd(&counts[i], counts_scratch[i]);
        }

        __global__ void averageCentroids(float *centroids, float *counts, int D)
        {
            int ind = blockIdx.x * blockDim.x + threadIdx.x;
            centroids[ind] /= counts[ind / D];
        }

        __global__ void checkConvergence(float *old_c, float *new_c, float *conv, int K, int D)
        {
            __shared__ float accum[1];
            
            int bid = blockIdx.x;
            int tid = threadIdx.x; 

            if (tid == 0) *accum = 0;
            __syncthreads();

            int index = bid * blockDim.x + tid; 

            if (index < K * D) {
                float diff  = old_c[index] - new_c[index];
                atomicAdd(accum, diff*diff);
            }

            __syncthreads();

            if (tid == 0) {
                atomicAdd(conv, *accum);
            }
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
        float *cuda_dataset, **cuda_centroids, *cuda_counts, *centroids[2], *cuda_conv;
        int *cuda_labels;

        CUDA_ERR(cudaMalloc(&cuda_dataset, ds_size));
        CUDA_ERR(cudaMalloc(&centroids[0], centroid_size));
        CUDA_ERR(cudaMalloc(&centroids[1], centroid_size));
        CUDA_ERR(cudaMalloc(&cuda_centroids, sizeof(float *) * 2));
        CUDA_ERR(cudaMalloc(&cuda_counts, sizeof(float) * K));
        CUDA_ERR(cudaMalloc(&cuda_labels, sizeof(int) * N));
        CUDA_ERR(cudaMalloc(&cuda_conv, sizeof(float)));

        // copy dataset to device
        CUDA_ERR(cudaMemcpy(cuda_dataset, ds.vecs, ds_size,
                            cudaMemcpyHostToDevice));
        CUDA_ERR(cudaMemcpy(centroids[0], l.centroids, centroid_size,
                            cudaMemcpyHostToDevice));
        CUDA_ERR(cudaMemcpy(cuda_centroids, centroids, sizeof(float *) * 2,
                            cudaMemcpyHostToDevice));

        const size_t N_THREADS = 1024;
        dim3 threads(N_THREADS);
        dim3 blocks((N + N_THREADS - 1) / N_THREADS);
        dim3 blocks_conv((K*D + N_THREADS - 1) / N_THREADS);

        std::vector<double> ms_per_iter;
        int iter = 0;
        while (iter++ < options.max_iters)
        {
            TIME_EXEC(ms_per_iter, {
                
                // check for convergence
                CUDA_ERR(cudaMemset(cuda_conv, 0, sizeof(float)));
                kcuda::checkConvergence<<<blocks_conv, threads>>>(centroids[0], centroids[1], cuda_conv, K, D);
                CUDA_ERR(cudaPeekAtLastError());
                CUDA_ERR(cudaDeviceSynchronize());

                float conv;
                CUDA_ERR(cudaMemcpy(&conv, cuda_conv, sizeof(float), cudaMemcpyDeviceToHost));

                if (std::sqrt(conv) <= options.threshold){
                    goto done; 
                }

                // zero second buffer and counts
                CUDA_ERR(cudaMemset(centroids[iter % 2], 0, centroid_size));
                CUDA_ERR(cudaMemset(cuda_counts, 0, sizeof(float) * K));
                CUDA_ERR(cudaDeviceSynchronize());

                if (options.gpu)
                    kcuda::labelVectors<<<blocks, threads>>>(
                        iter,
                        cuda_dataset, cuda_centroids,
                        cuda_counts, cuda_labels,
                        N, K, D);
                else if (options.gpu_shmem)
                    kcuda::labelVectorsShared<<<blocks, threads, sizeof(float) * (2 * D + 1) * K>>>(
                        iter,
                        cuda_dataset, cuda_centroids,
                        cuda_counts, cuda_labels,
                        N, K, D);

                CUDA_ERR(cudaPeekAtLastError());
                CUDA_ERR(cudaDeviceSynchronize());

                kcuda::averageCentroids<<<1, K * D>>>(centroids[iter % 2], cuda_counts, D);
                CUDA_ERR(cudaPeekAtLastError());
                CUDA_ERR(cudaDeviceSynchronize());
            })
        }
done:
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