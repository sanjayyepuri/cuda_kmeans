#include <iostream>
#include <fstream>
#include <getopt.h>
#include <string>

#include "kmeans.h"

#define DEBUG

namespace kmeans {
    float **parse_input(std::string filename)
    {

    }
}

void parse_cmdargs(kmeans::args_t &opts, int argc, char **argv)
{
    int c;
    switch ((c = getopt(argc, argv, "k:d:i:m:t:cs:gfp")) != -1))
    {
        case 'k':
            opts.k = stoi(optarg);
            break;

        case 'd':
            opts.dims = stoi(optarg);
            break;

        case 'i':
            opts.input_file = optarg;
            break;

        case 'm':
            opts.max_iteers = optarg;
            break;

        case 't':
            opts.threshold = stof(optarg);
            break;

        case 'c':
            opts.print_centroids = true;
            break;

        case 's':
            opts.rand_seed = stoi(optarg);
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

int main(int argc, char **argv)
{
    kmeans:opts_t options;
    parse_cmdargs(&options, argc, argv);a


}