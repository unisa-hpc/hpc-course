// tree_reduction_openmp.cpp
#include <iostream>
#include <random>
#include <omp.h>

double tree_reduce(const double *a, size_t n) {
    int num_threads;
    double *partial;
    double result = 0.0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double local_sum = 0.0;

        // Local sum for each thread
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            local_sum += a[i];
        }

        // Allocate partial array for thread-local sums
        #pragma omp single
        {
            num_threads  = omp_get_num_threads();
            partial = new double[num_threads];
            if (!partial) {
                std::cerr << "Alloc error" << std::endl;
                exit(1);
            }
        }

        // write the local sum into the partial array
        partial[tid] = local_sum;

        // wait for all threads to finish writing
        #pragma omp barrier

        // tree reduction
        for (int stride = 1; stride < num_threads; stride *= 2) {
            #pragma omp for schedule(static)
            for (int t = 0; t < num_threads - stride; t += 2 * stride) {
                partial[t] += partial[t + stride];
            }
            #pragma omp barrier
        }
        // wait for all threads to finish the reduction
        #pragma omp single
        {
            result = partial[0];
            delete[] partial;
            partial = nullptr;
        }
    }

    return result;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    size_t N = std::stoul(argv[1]);
    if (N == 0) {
        std::cerr << "Invalid size" << std::endl;
        return 1;
    }

    double *data = new double[N];
    if (!data) {
        std::cerr << "Alloc error" << std::endl;
        return 1;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads" << std::endl;

    for (size_t i = 0; i < N; ++i)
        data[i] = dis(gen);

    double sum = tree_reduce(data, N);
    std::cout << "Sum = " << sum << std::endl;

    delete[] data;
    return 0;
}