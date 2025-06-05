#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using DATA_TYPE = float;

const DATA_TYPE FLOAT_N = 3214212.01f;
const DATA_TYPE EPS = 0.005f;

void init_arrays(std::vector<DATA_TYPE>& data, int N) {
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            data[i * (N + 1) + j] = static_cast<DATA_TYPE>(i * j) / (N + 1);
        }
    }
}

void correlation(int N) {
    int size = N + 1;
    std::vector<DATA_TYPE> data(size * size);
    std::vector<DATA_TYPE> mean(size);
    std::vector<DATA_TYPE> stddev(size);
    std::vector<DATA_TYPE> symmat(size * size);

    init_arrays(data, N);

    // Compute mean of each column
    for (int j = 1; j <= N; ++j) {
        mean[j] = 0.0f;
        for (int i = 1; i <= N; ++i) {
            mean[j] += data[i * size + j];
        }
        mean[j] /= FLOAT_N;
    }

    // Compute stddev of each column
    for (int j = 1; j <= N; ++j) {
        stddev[j] = 0.0f;
        for (int i = 1; i <= N; ++i) {
            DATA_TYPE diff = data[i * size + j] - mean[j];
            stddev[j] += diff * diff;
        }
        stddev[j] /= FLOAT_N;
        stddev[j] = std::sqrt(stddev[j]);
        if (stddev[j] <= EPS) stddev[j] = 1.0f;
    }

    // Center and normalize the data
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            data[i * size + j] -= mean[j];
            data[i * size + j] /= std::sqrt(FLOAT_N);
            data[i * size + j] /= stddev[j];
        }
    }

    // Compute correlation matrix
    for (int j1 = 1; j1 <= N - 1; ++j1) {
        symmat[j1 * size + j1] = 1.0f;
        for (int j2 = j1 + 1; j2 <= N; ++j2) {
            symmat[j1 * size + j2] = 0.0f;
            for (int i = 1; i <= N; ++i) {
                symmat[j1 * size + j2] += data[i * size + j1] * data[i * size + j2];
            }
            symmat[j2 * size + j1] = symmat[j1 * size + j2];
        }
    }

    symmat[N * size + N] = 1.0f;

    std::cout << "Correlation matrix:\n";
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            std::cout << std::fixed << std::setprecision(2)
                      << symmat[i * size + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    int N = 5;  // Size of the matrix
    correlation(N);
    return 0;
}