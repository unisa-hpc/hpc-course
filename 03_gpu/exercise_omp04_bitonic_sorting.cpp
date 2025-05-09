// bitonic_sort_openmp.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>

// In-place bitonic sort on array 'a' of length N (must be a power of two)
void bitonic_sort(float *a, int N) {
    #pragma omp parallel
    {
        // Build bitonic sequences of length k = 2, 4, 8, ..., N
        for (int k = 2; k <= N; k <<= 1) {
            // For each subsequence, do the bitonic merge step with stride j
            for (int j = k >> 1; j > 0; j >>= 1) {
                #pragma omp for schedule(static)
                for (int i = 0; i < N; ++i) {
                    int ixj = i ^ j;
                    if (ixj > i) {
                        // Determine sort direction based on the k-bit of i
                        bool ascending = ((i & k) == 0);
                        if ((ascending  && a[i] >  a[ixj]) ||
                            (!ascending && a[i] <  a[ixj])) {
                            std::swap(a[i], a[ixj]);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int N = 1 << 20; // e.g., 1 million elements (must be a power of two)
    std::vector<float> data(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1'000'000);

    for (int i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    // Perform bitonic sort
    bitonic_sort(data.data(), N);

    // Quick verification that the array is sorted
    for (int i = 1; i < N; ++i) {
        if (data[i-1] > data[i]) {
            std::cerr << "Sort failed at index " << i << "\n";
            return 1;
        }
    }
    std::cout << "Bitonic sort completed successfully.\n";
    return 0;
}