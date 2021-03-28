
#include "kmeans.h"

namespace kmeans 
{
    void copyCentroids(const Dataset &ds, float *centroids)
    {
        for (size_t c : ds.init_centroids)
        {
            for (int i = 0; i < ds.dims; ++i)
            {
                size_t k = I(c, i, ds.dims);
                centroids[k] = ds.vecs[k];
            }
        }
    }

    void kmeansSequential(const Dataset &ds)
    {
        float *centroids[2];

        centroids[0] = new float[ds.n * ds.dims];
        centroids[1] = new float[ds.n * ds.dims];

        // initialize the first centroid scratch area
        copyCentroids(ds, centroids[0]);
    }    
}