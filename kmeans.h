#pragma once

#include <vector>

#define BSTR(b) (b ? "true" : "false")
#define DEBUG

#define I(r, c, D) (D*r + c)

namespace kmeans
{
    struct Args {
        int k = 0;
        int dims = 0;
        int max_iters = 20;
        int rand_seed = 0;

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
}


