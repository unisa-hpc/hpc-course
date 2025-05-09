#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>

using DATA_TYPE = float;
constexpr DATA_TYPE float_n = 3214212.01f;

// Initialize data[i][j] = (i * j) / M, for i,j = 0..M-1.
// We size the array (M+1)x(M+1) so that we can index from 1..M in covariance().
void init_arrays(DATA_TYPE* data, size_t M) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
            data[i * (M+1) + j] = (static_cast<DATA_TYPE>(i) * j) / static_cast<DATA_TYPE>(M);
        }
    }
}

// Compute:
// 1) mean[j] = (1/float_n) * Σ_{i=1..N} data[i][j]
// 2) data[i][j] -= mean[j]
// 3) symmat[j1][j2] = Σ_{i=1..N} data[i][j1] * data[i][j2], mirrored to make it symmetric.
void covariance(DATA_TYPE* data,
                DATA_TYPE* symmat,
                DATA_TYPE* mean,
                size_t M)
{
    const size_t N = M;

    // column means
    for (size_t j = 1; j <= M; ++j) {
        mean[j] = 0.0f;
        for (size_t i = 1; i <= N; ++i) {
            mean[j] += data[i * (M+1) + j];
        }
        mean[j] /= float_n;
    }

    // center columns
    for (size_t i = 1; i <= N; ++i) {
        for (size_t j = 1; j <= M; ++j) {
            data[i * (M+1) + j] -= mean[j];
        }
    }

    // covariance matrix (upper triangle + mirror)
    for (size_t j1 = 1; j1 <= M; ++j1) {
        for (size_t j2 = j1; j2 <= M; ++j2) {
            DATA_TYPE sum = 0.0f;
            for (size_t i = 1; i <= N; ++i) {
                sum += data[i * (M+1) + j1] * data[i * (M+1) + j2];
            }
            symmat[j1 * (M+1) + j2] = sum;
            symmat[j2 * (M+1) + j1] = sum;
        }
    }
}

int main(int argc, char** argv) {
    size_t M = 5;
    if (argc > 1) {
        M = std::atoi(argv[1]);
        if (M < 1) M = 5;
    }

    std::vector<DATA_TYPE> data((M+1)*(M+1));
    std::vector<DATA_TYPE> mean(M+1);
    std::vector<DATA_TYPE> symmat((M+1)*(M+1));

    init_arrays(data.data(), M);
    covariance(data.data(), symmat.data(), mean.data(), M);

    std::cout << "Covariance matrix (" << M << "×" << M << "):\n";
    for (size_t i = 1; i <= M; ++i) {
        for (size_t j = 1; j <= M; ++j) {
            std::cout << std::fixed << std::setprecision(4)
                      << symmat[i*(M+1) + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}