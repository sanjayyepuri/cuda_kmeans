#pragma once

#include <string>

namespace kmeans
{
    struct args_t {
        int k;
        int dims;
        int max_iters;
        int rand_seed; 

        std::string input_file;
        float threshold; 

        bool print_centroids;
        bool gpu;
        bool gpu_shmem;
        bool kmeans_pp; 
    };

    // read input file and return 2d array of vectors
    float **parse_input(std::string filename);



    
}


