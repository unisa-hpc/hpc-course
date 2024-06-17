#include <sycl/sycl.hpp>
#include <vector>

constexpr size_t SIZE = 1024;
constexpr size_t LOCAL_SIZE = 16;

void print_matrix(std::vector<int>& vec, size_t width, size_t height) {
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      std::cout << vec[j * width + i] << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  std::vector<int> inA(SIZE * SIZE, 2);
  std::vector<int> inB(SIZE * SIZE, 2);
  std::vector<int> out(SIZE * SIZE, 0);

  sycl::queue q{ sycl::gpu_selector_v };

  sycl::buffer<int, 2> inA_buf{ inA.data(), sycl::range<2>{SIZE, SIZE} };
  sycl::buffer<int, 2> inB_buf{ inB.data(), sycl::range<2>{SIZE, SIZE} };
  sycl::buffer<int, 2> out_buf{ out.data(), sycl::range<2>{SIZE, SIZE} };

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor inA_acc{inA_buf, cgh, sycl::read_only};
    sycl::accessor inB_acc{inB_buf, cgh, sycl::read_only};
    sycl::accessor out_acc{out_buf, cgh, sycl::write_only, sycl::no_init};

    sycl::local_accessor<int, 2> tileA {sycl::range<2>{LOCAL_SIZE, LOCAL_SIZE}, cgh};
    sycl::local_accessor<int, 2> tileB {sycl::range<2>{LOCAL_SIZE, LOCAL_SIZE}, cgh};

    size_t hA = SIZE;
    size_t wB = SIZE;
    size_t commSide = SIZE;
    size_t localSize = LOCAL_SIZE;

    cgh.parallel_for(sycl::nd_range<2>{{hA, wB}, {localSize, localSize}}, [=](sycl::nd_item<2> item) {
      uint gidX = item.get_global_id(0);
      uint gidY = item.get_global_id(1);
      uint lidX = item.get_local_id(0);
      uint lidY = item.get_local_id(1);

      uint num_tiles = (hA * wB) / (localSize * localSize);
      int sum = 0;
      for (int i = 0; i < num_tiles; i++) {
        tileA[lidX][lidY] = inA_acc[gidX][i * localSize + lidY];
        tileB[lidX][lidY] = inB_acc[i * localSize + lidX][gidY];
        sycl::group_barrier(item.get_group());
        for (int j = 0; j < localSize; j++) {
          sum += tileA[lidX][j] * tileB[j][lidY];
        }
        sycl::group_barrier(item.get_group());
      }
      out_acc[gidX][gidY] = sum;
    });
  });
  q.wait_and_throw();

  out_buf.get_host_access(); // only for triggering the write back
  print_matrix(out, 16, 16); // print only few elements
}
