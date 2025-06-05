#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>

using DATA_TYPE = float;

// Fill A[i][j] = ((i+1)*(j+1)) / (N+1)
void init_array(std::vector<DATA_TYPE>& A, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A[i * N + j] = (static_cast<DATA_TYPE>(i + 1) * (j + 1)) / static_cast<DATA_TYPE>(N + 1);
        }
    }
}

// Classical Gram–Schmidt: given A (NxN), compute Q and R so that A = Q * R,
// Q has orthonormal columns, R is upper-triangular.
void gramschmidt(const std::vector<DATA_TYPE>& A_in,
                 std::vector<DATA_TYPE>& Q,
                 std::vector<DATA_TYPE>& R,
                 size_t N)
{
    // Make a working copy of A, since we overwrite its columns
    std::vector<DATA_TYPE> A = A_in;

    for (size_t k = 0; k < N; ++k) {
        // Compute the norm of column k of A
        DATA_TYPE nrm = 0;
        for (size_t i = 0; i < N; ++i) {
            nrm += A[i * N + k] * A[i * N + k];
        }
        R[k * N + k] = std::sqrt(nrm);

        // Normalize column k → Q[:,k]
        for (size_t i = 0; i < N; ++i) {
            Q[i * N + k] = A[i * N + k] / R[k * N + k];
        }

        // For each remaining column j>k:
        //   - compute projection R[k,j] = Q[:,k]·A[:,j]
        //   - subtract that component: A[:,j] -= Q[:,k] * R[k,j]
        for (size_t j = k + 1; j < N; ++j) {
            R[k * N + j] = 0;
            for (size_t i = 0; i < N; ++i) {
                R[k * N + j] += Q[i * N + k] * A[i * N + j];
            }
            for (size_t i = 0; i < N; ++i) {
                A[i * N + j] -= Q[i * N + k] * R[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    size_t N = 5;
    if (argc > 1) {
        int v = std::atoi(argv[1]);
        if (v > 0) N = static_cast<size_t>(v);
    }

    std::vector<DATA_TYPE> A(N * N), Q(N * N, 0.0f), R(N * N, 0.0f);

    init_array(A, N);

    gramschmidt(A, Q, R, N);

    std::cout << "R =\n";
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << R[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nQ =\n";
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4)
                      << Q[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}