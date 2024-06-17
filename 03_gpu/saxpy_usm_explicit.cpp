#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
  const size_t size = 10000;

  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  // USM allocation, explicit data movement
  float* x_data = static_cast<float*>(malloc(size * sizeof(float)));
  float* y_data = static_cast<float*>(malloc(size * sizeof(float)));
  // data initialization
  for (size_t i = 0; i < size; i++)  x_data[i] = 1.0f;
  for (size_t i = 0; i < size; i++)  y_data[i] = 2.0f;
  float a = 0.5;
  // USM allocation
  float* x_device = malloc_device<float>(size, q);
  float* y_device = malloc_device<float>(size, q);
  // copy from host to device
  q.memcpy(x_device, x_data, size * sizeof(float));
  q.memcpy(y_device, y_data, size * sizeof(float)).wait();
  q.submit([&](handler& h) {
    h.parallel_for(range<1>{size}, [=](item<1> i) {
      y_device[i] = a * x_device[i] + y_device[i];
      });
    }).wait();
  // copy from device to host
  q.memcpy(x_data, x_device, size * sizeof(float));
  q.memcpy(y_data, y_device, size * sizeof(float)).wait();
  // Print the first 20 results of saxpy.
  std::cout << "\ny: ";
  for (size_t i = 0; i < 20; i++) std::cout << "" << y_data[i] << " ";
  std::cout << "\nsaxpy_usm EXPLICIT successfully completed on device\n";
  free(x_device, q);
  free(y_device, q);
  free(x_data);
  free(y_data);
  return 0;
}
