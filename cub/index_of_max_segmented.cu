#include <cub/cub.cuh>
#include <cstdlib>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

#define cudaSucceeded(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) {
    std::cerr << "cudaAssert failed: "
              << cudaGetErrorString(code)
              << file << ":" << line
              << std::endl;
    if (abort) {
      exit(code);
    }
  }
}

static struct timeval t_start, t_end;

void start_timing() {
  cudaSucceeded(cudaDeviceSynchronize());
  gettimeofday(&t_start, NULL);
}

void end_timing() {
  cudaSucceeded(cudaDeviceSynchronize());
  gettimeofday(&t_end, NULL);
}

int get_us() {
  return (t_end.tv_sec*1000000+t_end.tv_usec) - (t_start.tv_sec*1000000+t_start.tv_usec);
}

int main(int argc, char** argv) {

  int num_segments = pow(2,atoi(argv[1]));
  int segment_size = pow(2,atoi(argv[2]));
  int num_elements = num_segments * segment_size;

  cerr << num_segments << " segments of " << segment_size << " elements each" << endl;

  int *h_offsets = new int[num_segments+1];

  for (int i = 0; i < num_segments+1; i++) {
    h_offsets[i] = i * segment_size;
  }

  int *h_in = new int[num_elements];

  for (int i = 0; i < num_elements; i++) {
    h_in[i] = i % segment_size;
  }

  int *d_offsets;
  int *d_in;
  cub::KeyValuePair<int, int> *d_out;
  cudaSucceeded(cudaMalloc(&d_offsets, (num_segments+1)*sizeof(int)));
  cudaSucceeded(cudaMalloc(&d_in, num_elements*sizeof(int)));
  cudaSucceeded(cudaMalloc(&d_out, num_segments*sizeof(cub::KeyValuePair<int, int>)));

  cudaSucceeded(cudaMemcpy(d_offsets, h_offsets, (num_segments+1)*sizeof(int),
                           cudaMemcpyHostToDevice));
  cudaSucceeded(cudaMemcpy(d_in, h_in, num_elements*sizeof(int),
                           cudaMemcpyHostToDevice));
  cudaSucceeded(cudaDeviceSynchronize());

  void     *d_temp_storage = NULL;

  // Now time.

  static const int num_runs = 100;
  int total_us = 0;

  size_t temp_storage_bytes = 0;
  cudaSucceeded(cub::DeviceSegmentedReduce::ArgMax
                (d_temp_storage, temp_storage_bytes, d_in, d_out,
                 num_segments,
                 d_offsets, d_offsets + 1));
  cudaSucceeded(cudaMalloc(&d_out, num_segments*sizeof(int)));
  cudaSucceeded(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  for (int i = 0; i < num_runs; i++) {
    start_timing();
    cudaSucceeded(cub::DeviceSegmentedReduce::ArgMax
                  (d_temp_storage, temp_storage_bytes, d_in, d_out,
                   num_segments,
                   d_offsets, d_offsets + 1));
    end_timing();
    total_us += get_us();
  }
  cerr << total_us/num_runs << "us" << endl;
  if (!isatty(1)) {
    cout << total_us/num_runs;
  }

  // No checking for this one; CUB seems like a trustworthy sort.
  return 0;
}
