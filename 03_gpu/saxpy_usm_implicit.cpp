#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
    const size_t size = 10000;

    queue q;
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    // USM allocation, implicit data movement
    float* x = malloc_shared<float>(size, q);
    float* y = malloc_shared<float>(size, q);
    // data initialization
    for (size_t i = 0; i < size; i++)  x[i] = 1.0f;
    for (size_t i = 0; i < size; i++)  y[i] = 2.0f;
    float a = 0.5;
    range<1> num_items{ size };
    q.submit([&](handler& h) {
        h.parallel_for(num_items, [=](item<1> i) {
            y[i] = a * x[i] + y[i];
            });
        });
    q.wait();
    // Print the first 20 results of saxpy.
    std::cout << "\ny: ";
    for (size_t i = 0; i < 20; i++) std::cout << "" << y[i] << " ";
    std::cout << "\nsaxpy_usm IMPLICIT successfully completed on device.\n";
    free(x, q);
    free(y, q);
    return 0;
}
