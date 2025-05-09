#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using DATA_TYPE = double;
constexpr size_t TMAX = 500;

// We allocate ex as NX×(NY+1) and ey as (NX+1)×NY to match the original strides.
void init_arrays(DATA_TYPE* fict,
                 DATA_TYPE* ex,
                 DATA_TYPE* ey,
                 DATA_TYPE* hz,
                 size_t NX)
{
    const size_t NY = NX;

    // time-series “boundary” array
    for (size_t t = 0; t < TMAX; ++t) {
        fict[t] = static_cast<DATA_TYPE>(t);
    }

    // 2D fields
    for (size_t i = 0; i < NX; ++i) {
        for (size_t j = 0; j < NY; ++j) {
            ex[i * (NY + 1) + j] = ((DATA_TYPE)i * (j + 1) + 1) / static_cast<DATA_TYPE>(NX);
            ey[i * NY       + j] = ((DATA_TYPE)(i - 1) * (j + 2) + 2) / static_cast<DATA_TYPE>(NX);
            hz[i * NY       + j] = ((DATA_TYPE)(i - 9) * (j + 4) + 3) / static_cast<DATA_TYPE>(NX);
        }
        // We leave the “extra” column in ex[i][NY] and “extra” row ey[NX][*] uninitialized;
        // they are never read in the core loops.
    }
}

// Run the FDTD update for TMAX timesteps:
//   ey[0][j] = fict[t]
//   ey[i>0][j] -= 0.5*(hz[i][j] − hz[i−1][j])
//   ex[i][j>0] -= 0.5*(hz[i][j] − hz[i][j−1])
//   hz[i][j] -= 0.7*(ex[i][j+1] − ex[i][j] + ey[i+1][j] − ey[i][j])
void runFdtd(DATA_TYPE* fict,
             DATA_TYPE* ex,
             DATA_TYPE* ey,
             DATA_TYPE* hz,
             size_t NX)
{
    const size_t NY = NX;
    for (size_t t = 0; t < TMAX; ++t) {
        // update ey at i=0 from the “fict” source
        for (size_t j = 0; j < NY; ++j) {
            ey[0 * NY + j] = fict[t];
        }
        // update ey for i=1..NX-1
        for (size_t i = 1; i < NX; ++i) {
            for (size_t j = 0; j < NY; ++j) {
                ey[i * NY + j] -= 0.5 * (hz[i * NY + j] - hz[(i - 1) * NY + j]);
            }
        }
        // update ex for j=1..NY-1
        for (size_t i = 0; i < NX; ++i) {
            for (size_t j = 1; j < NY; ++j) {
                ex[i * (NY + 1) + j] -= 0.5 * (hz[i * NY + j] - hz[i * NY + (j - 1)]);
            }
        }
        // update hz everywhere
        for (size_t i = 0; i < NX; ++i) {
            for (size_t j = 0; j < NY; ++j) {
                hz[i * NY + j] -= 0.7 * (
                    ex[i * (NY + 1) + (j + 1)] - ex[i * (NY + 1) + j]
                  + ey[(i + 1) * NY + j]      - ey[i * NY + j]
                );
            }
        }
    }
}

int main(int argc, char** argv) {
    size_t NX = 500;
    if (argc > 1) {
        int v = std::atoi(argv[1]);
        if (v > 0) NX = static_cast<size_t>(v);
    }

    const size_t NY = NX;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<DATA_TYPE> fict(TMAX);
    std::vector<DATA_TYPE> ex (NX * (NY + 1));
    std::vector<DATA_TYPE> ey ((NX + 1) * NY);
    std::vector<DATA_TYPE> hz (NX * NY);

    init_arrays(fict.data(), ex.data(), ey.data(), hz.data(), NX);
    runFdtd   (fict.data(), ex.data(), ey.data(), hz.data(), NX);

    size_t ci = NX/2, cj = NY/2;
    std::cout << "hz["<<ci<<"]["<<cj<<"] = "
              << std::fixed << std::setprecision(6)
              << hz[ci * NY + cj] << "\n";

    return 0;
}