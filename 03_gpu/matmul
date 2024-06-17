#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
  const size_t size = 256;
  std::vector<float> A_mat(size * size, 1.0f);
  std::vector<float> B_mat(size * size, 2.0f);
  std::vector<float> C_mat(size * size, 0.0f);

  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  range<2> mat_size(size, size);
  buffer<float, 2> A_buf(A_mat.data(), mat_size);
  buffer<float, 2> B_buf(B_mat.data(), mat_size);
  buffer<float, 2> C_buf(C_mat.data(), mat_size);

  q.submit([&](handler& h) {
    accessor A(A_buf, h, read_only);
    accessor B(B_buf, h, read_only);
    accessor C(C_buf, h, read_write);
    h.parallel_for(mat_size, [=](item<2> id) {
      C[id] = 0;
      for (size_t k = 0; k < size; k++)
        C[id] += A[{id[0], k}] * B[{k, id[1]}];
    });
  });
  host_accessor C_res(C_buf, read_only);

  // Print the first 5x5 results of matmul.
  std::cout << "\nC:\n";
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++)
      std::cout << C_res[{i, j}] << " ";
    std::cout << "\n";
  }
  std::cout << "\nmatmul successfully completed\n";
  return 0;
}
