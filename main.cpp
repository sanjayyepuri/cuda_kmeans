#include <iostream>
#include <fstream>
#include <getopt.h>
#include <string>

#include "kmeans.h"

void parseCmdArgs(kmeans::Args &opts, int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "k:d:i:m:t:cs:gfp")) != -1)
    {
        switch (c) {
            case 'k':
                opts.k = std::stoi(optarg);
                break;

            case 'd':
                opts.dims = std::stoi(optarg);
                break;

            case 'i':
                opts.input_file = optarg;
                break;

            case 'm':
                opts.max_iters = std::stoi(optarg);
                break;

            case 't':
                opts.threshold = std::stof(optarg);
                break;

            case 'c':
                opts.print_centroids = true;
                break;

            case 's':
                opts.rand_seed = std::stoi(optarg);
                break;

            case 'g':
                opts.gpu = true;
                break;

            case 'f':
                opts.gpu_shmem = true;
                break;

            case 'p':
                opts.kmeans_pp = true;
                break;

            default:
                abort();
                break;
        }
    }
}

void printVectors(kmeans::Dataset &ds)
{
    for (int r = 0; r < ds.n; ++r)
    {
        for (int c = 0; c < ds.dims; ++c)
            std::cout << ds.vecs[I(r, c, ds.dims)] << " ";
        std::cout << std::endl;
    }
}

void printIntialCentroids(kmeans::Dataset &ds)
{
    for (int c : ds.init_centroids)
    {
        std::cout << c << " ";
        for (int i = 0; i < ds.dims; ++i) 
            std::cout << ds.vecs[I(c, i, ds.dims)] << " ";
        std::cout << std::endl;
    }
}


int main(int argc, char **argv)
{
    kmeans::Args options;
    parseCmdArgs(options, argc, argv);

    // seed random numbers
    srand(options.rand_seed);

#ifdef DEBUG
    std::cout << "num clusters: " << options.k << std::endl;
    std::cout << "dimensions: " << options.dims << std::endl;
    std::cout << "input file: " << options.input_file << std::endl;
    std::cout << "max iterations: " << options.max_iters << std::endl;
    std::cout << "threshold: " << options.threshold << std::endl;
    std::cout << "print centroids: " << BSTR(options.print_centroids) << std::endl;
    std::cout << "random seed: " << options.rand_seed << std::endl;
    std::cout << "gpu: " << BSTR(options.gpu) << std::endl;
    std::cout << "gpu shared memory: " << BSTR(options.gpu_shmem) << std::endl;
    std::cout << "kmeans++: " << BSTR(options.kmeans_pp) << std::endl;
#endif 
    
    kmeans::Dataset ds = kmeans::buildDataset(options); 

#ifdef DEBUG 
    // printVectors(ds);
    printIntialCentroids(ds);
#endif 

    return 0;
}