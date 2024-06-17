#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main(int, char**) {
  constexpr size_t size = 1024;
  std::vector<float> A_mat(size * size, 1.0f);
  std::vector<float> B_mat(size * size, 2.0f);
  std::vector<float> C_mat(size * size, 0.0f);

  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
  range<2> mat_size(size, size);
  buffer<float, 2> A_buf(A_mat.data(), mat_size);
  buffer<float, 2> B_buf(B_mat.data(), mat_size);
  buffer<float, 2> C_buf(C_mat.data(), mat_size);

  constexpr int tile_size = 4;
  q.submit([&](handler& h) {
    accessor A(A_buf, h, read_only);
    accessor B(B_buf, h, read_only);
    accessor C(C_buf, h, write_only);
    h.parallel_for(nd_range<2>{{size, size}, { 1, tile_size }}, 
      [=](nd_item<2> id) {
      auto sg = id.get_sub_group();
      size_t  i = id.get_global_id()[0];
      size_t  j = id.get_global_id()[1];
      size_t  l = id.get_local_id()[1];
      float sum = 0.f;
      for (size_t t = 0; t < size; t += tile_size) {
        float tileA = A[{i, t + l}]; // load a 1D tile
        for (size_t k = 0; k < tile_size; k++)
          sum += group_broadcast(sg, tileA, k) * B[{t + k, j}];
      }
      C[{i,j}] = sum;
    });
  });
  host_accessor C_res(C_buf, read_only);

  // Print the first 5x5 results of matmul
  std::cout << "\nC:\n";
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 5; j++)
      std::cout << C_res[{i, j}] << " ";
    std::cout << "\n";
  }
  std::cout << "\nmatmul_bcast successfully completed\n";
  return 0;
}
