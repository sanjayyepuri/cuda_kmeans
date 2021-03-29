#include <math.h>
#include <cstring>
#include <iostream>

#include "kmeans.h"

namespace kmeans
{
    float avgTimeMS(std::vector<double> ms_per_iter)
    {
        float t = 0;
        for (float ms : ms_per_iter)
        {
            t += ms; 
        }

        return t / ms_per_iter.size();
    }

    void copyCentroids(const Dataset &ds, float *centroids)
    {
        for (int c = 0; c < ds.init_centroids.size(); ++c)
        {
            for (int i = 0; i < ds.dims; ++i)
            {
                size_t cid = ds.init_centroids[c];
                centroids[I(c, i, ds.dims)] = ds.vecs[I(cid, i, ds.dims)];
            }
        }
    }
    
    // compute the square distance between two vectors 
    float distance(float *a, float *b, const size_t dims)
    {
        float dist = 0;
        for (int i = 0; i < dims; ++i)
        {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return dist; 
    }

    void add(float *a, float *b, const size_t dims)
    {
        for (int i = 0; i < dims; ++i)
            a[i] += b[i];
    }

    bool convergence(float *o_centroids, float *n_centroids, float threshold, int k, int dims)
    {
        int N = k * dims;
        float dist = distance(o_centroids, n_centroids, N);

        return std::sqrt(dist) <= threshold;
    }

    inline void zeroBuf(float *v, const size_t dims)
    {
        for (int i = 0; i < dims; ++i)
            v[i] = 0;
    }

    Labels kmeansSequential(const Dataset &ds, const Args &options)
    {
        size_t N = ds.n, D = ds.dims, K = ds.k;

        float *centroids[2];
        centroids[0] = new float[K * D];
        centroids[1] = new float[K * D];

        // initialize the first centroid scratch area
        copyCentroids(ds, centroids[0]);

        // store the labels for each of the
        int *labels = new int[N];

        std::vector<double> ms_per_iter;

        int iters = 0;
        while (iters++ < options.max_iters 
            && !convergence(centroids[0], centroids[1], options.threshold, K, D))
        {
            TIME_EXEC(ms_per_iter,
            {
                int cc = (iters-1) % 2; 
                int nc = iters % 2;

                zeroBuf(centroids[nc], K * D);

                int counts[K] = {0}; // tracks the num points in each cluster
                for (int x = 0; x < N; ++x)
                {
                    // label each point
                    float *x_vec = &ds.vecs[I(x, 0, D)];
                    
                    float min_dist = MAXFLOAT;
                    size_t centroid_id = -1;

                    for (int c = 0; c < K; ++c)
                    {
                        float *centroid = &centroids[cc][I(c, 0, D)];
                        float dist = distance(centroid, x_vec, D);

                        if (dist < min_dist) {
                            min_dist = dist; 
                            centroid_id = c;
                        }
                    }

                    labels[x] = centroid_id;

                    // accumulate sums to compute mean 
                    add(&centroids[nc][I(centroid_id, 0, D)], x_vec, D);
                    counts[centroid_id]++;
                }

                // compute new centroids
                for (int r = 0; r < K; ++r)
                    for (int c = 0; c < D; ++c)
                        centroids[nc][I(r, c, D)] /= (float) counts[r];
            })
        }

        float time_per_iter = avgTimeMS(ms_per_iter);
        printf("%d, %lf\n", (int) ms_per_iter.size(), time_per_iter);

        Labels l;
        l.centroids = centroids[0]; // 0 and 1 are within a threshold
        l.labels = labels;

        // cleanup 
        delete[] centroids[1];
        return l;
    }
}