#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
  const size_t size = 128;

  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  range<2> mat_size(size, size);
  buffer<float, 2> A_buf(mat_size);
  buffer<float, 2> B_buf(mat_size);
  buffer<float, 2> C_buf(mat_size);
  buffer<float, 2> D_buf( mat_size);
  buffer<float, 2> temp_buf(mat_size);
  float alpha = 0.5f;

  // init A and B
  q.submit([&](handler& h) {
    accessor A(A_buf, h, write_only);
    accessor B(B_buf, h, write_only);
    h.parallel_for(mat_size, [=](item<2> id) {
      A[id] = 1.0f;
      B[id] = 2.0f;
    });
  });  
  // init C
  q.submit([&](handler& h) {
    accessor C(C_buf, h, write_only);
    h.parallel_for(mat_size, [=](item<2> id) {
      C[id] = 0.3f;
    });
  });
  // matmul temp = alpha*A*B
  q.submit([&](handler& h) {
      accessor A(A_buf, h, read_only);
      accessor B(B_buf, h, read_only);
      accessor temp(temp_buf, h, read_write);
      h.parallel_for(mat_size, [=](item<2> id) {
        temp[id] = 0;
        for (size_t k = 0; k < size; k++)
          temp[id] += alpha * A[{id[0], k}] * B[{k, id[1]}];
        });
      });
    // matmul D = temp * C
  q.submit([&](handler& h) {
    accessor temp(temp_buf, h, read_only);
    accessor C(C_buf, h, read_only);
    accessor D(D_buf, h, read_write);
    h.parallel_for(mat_size, [=](item<2> id) {
      D[id] = 0;
      for (size_t k = 0; k < size; k++)
        D[id] += temp[{id[0], k}] + C[{k, id[1]}];
      });
    });
  host_accessor D_res(D_buf, read_only);

  // Print the first 5x5 results of matmul.
  std::cout << "\nD:\n";
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++)
      std::cout << D_res[{i, j}] << " ";
    std::cout << "\n";
  }

  std::cout << "\n2matmul successfully completed\n";
  return 0;
}
