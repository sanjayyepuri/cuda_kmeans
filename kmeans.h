#pragma once

#include <string>

namespace kmeans
{
    struct args_t {
        int k;
        int dims;
        int max_iters;
        int rand_seed;

        char *input_file;
        float threshold;

        bool print_centroids = false;
        bool gpu = false;
        bool gpu_shmem = false;
        bool kmeans_pp = false;
    };

    // read input file and return 2d array of vectors
    float **parse_input(std::string filename);




}


