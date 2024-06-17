#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
  const size_t size = 100000;
  std::vector<float> x_vec(size, 1.0f);
  std::vector<float> y_vec(size, 2.0f);
  float a = 0.5;

  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  buffer x_buf(x_vec);
  buffer y_buf(y_vec);
  range<1> num_items{ x_vec.size() };
  q.submit([&](handler& h) {
    accessor x(x_buf, h, read_only);
    accessor y(y_buf, h, read_write);
    h.parallel_for(num_items, [=](item<1> i) {
      y[i] = a * x[i] + y[i];
      });
    });
  host_accessor y_res(y_buf, read_only);

  // Print the first 20 results of saxpy.
  std::cout << "\ny: ";
  for (size_t i = 0; i < 20; i++) std::cout << y_res[i] << " ";
  std::cout << "\nsaxpy successfully completed\n";
  return 0;
}
