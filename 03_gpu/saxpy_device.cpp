#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
  constexpr size_t size = 100000;
  std::vector<float> x_vec(size, 1.0f);
  std::vector<float> y_vec(size, 2.0f);
  constexpr float a = 0.5;

  // exception handling
  try {
    // selector logic
    //queue q(gpu_selector_v);
    //queue q(cpu_selector_v);
    //queue q(accelerator_selector_v);
    //queue q(default_selector_v);
    /*device d = device{aspect_selector(
      std::vector{aspect::fp16, aspect::gpu}, // allowed aspects
      std::vector{sycl::aspect::custom}  // disallowed aspects
    ) };
    queue q(d);*/
    std::vector<device> dv = device::get_devices(info::device_type::all);
    for (auto& di : dv) std::cout << "Found: " << di.get_info<info::device::name>() << "\n"; //switch device to auto&
    queue q(dv[0]); // pick the 1st GPU 
    std::cout << "Selected device: " << q.get_device().get_info<info::device::name>() << "\n";
    {
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
    } // Queue syncronize at buffer destruction

    // Print the first 20 results of saxpy.
    std::cout << "\ny: ";
    for (size_t i = 0; i < 20; i++) std::cout << y_vec[i] << " ";
    std::cout << "\nsaxpy_devices successfully completed\n";
  }
  catch (exception const& e) {
    std::cout << "An exception is caught for saxpy.\n";
  }
  return 0;
}
