#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
  constexpr size_t size = 1000000;
  std::vector<float> x_vec(size, 0.5f);
  std::vector<float> y_vec(size, 2.0f);
  float dot_value = 0.0f;

  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  buffer x_buf(x_vec);
  buffer y_buf(y_vec);
  buffer dot_buf(&dot_value, range(1));
  range<1> num_items{ x_vec.size() };
  q.submit([&](handler& h) {
    accessor x(x_buf, h, read_only);
    accessor y(y_buf, h, read_write);
    auto sum_reduction = reduction(dot_buf, h, plus<>());
    h.parallel_for(num_items, sum_reduction, 
      [=](item<1> i, auto &dot) {
        float product = x[i] * y[i];
        dot.combine(product);
      });
    });
  host_accessor dot_res(dot_buf, read_only);

  // Print the dot product
  std::cout << "\ndot product value: " << dot_res[0];
  std::cout << "\ndotproduct successfully completed\n";
  return 0;
}  
