#include <fstream>

#include "kmeans.h"

namespace kmeans {

    float rand_float()
    {
        return static_cast<float>(rand()) / static_cast<float>((long long) RAND_MAX + 1);
    }

    void kmeansInitCentroids(Dataset &ds)
    {
        for (int i = 0; i < ds.k; ++i)
        {
            size_t idx = (size_t) (rand_float() * ds.n); 
            ds.init_centroids.push_back(idx);
        }
    }

    void parseInput(Dataset &ds, char *filename)
    {
        std::ifstream input_s (filename);

        // read in number of vectors 
        int N;
        input_s >> N; 
        ds.n = N;

        // store the vectors in a chunk of memory
        ds.vecs = new float[N * ds.dims];

        for (int r = 0; r < N; ++r)
        {
            float f; 
            input_s >> f; // do we really need the vector id 

            for (int c = 0; c < ds.dims; ++c)
            {
                input_s >> f;
                ds.vecs[I(r, c, ds.dims)] = f; 
            }
        }
    }

    Dataset buildDataset(Args &options)
    {
        Dataset ds; 

        ds.dims = options.dims;
        ds.k = options.k;

        parseInput(ds, options.input_file);
        kmeansInitCentroids(ds);

        return ds;
    }
}

