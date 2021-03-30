#include <fstream>

#include "kmeans.h"

namespace kmeans
{

    float rand_float()
    {
        return static_cast<float>(rand()) / static_cast<float>((long long)RAND_MAX + 1);
    }

    void kmeansInitCentroids(Dataset &ds)
    {
        for (int i = 0; i < ds.k; ++i)
        {
            size_t idx = (size_t)(rand_float() * ds.n);
            ds.init_centroids.push_back(idx);
        }
    }

    void parseInput(Dataset &ds, char *filename)
    {
        std::ifstream input_s(filename);

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

    void printInitialCentroids(kmeans::Dataset &ds)
    {
        for (int c : ds.init_centroids)
        {
            std::cout << c << " ";
            for (int i = 0; i < ds.dims; ++i)
                std::cout << ds.vecs[I(c, i, ds.dims)] << " ";
            std::cout << std::endl;
        }
    }

    Dataset buildDataset(Args &options)
    {
        Dataset ds;

        ds.dims = options.dims;
        ds.k = options.k;

        parseInput(ds, options.input_file);

        if (options.kmeans_pp)
            kmeansppInitCentroids(ds);
        else
            kmeansInitCentroids(ds);

#ifdef DEBUG
        printInitialCentroids(ds);
#endif

        return ds;
    }

    void printTimeMs(std::vector<double> ms_per_iter)
    {
        float t = 0;
        for (float ms : ms_per_iter)
        {
            t += ms;
        }

        printf("%d, %lf\n", (int)ms_per_iter.size(), t / ms_per_iter.size());
    }
}
