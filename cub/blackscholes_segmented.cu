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

class make_option {
private:
  __host__ __device__ static
  float horner(float x) {
    float c1 = 0.31938153, c2 = -0.356563782, c3 = 1.781477937, c4 = -1.821255978, c5 = 1.330274429;
    return x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5))));
  }


  __host__ __device__ static
  float cnd0(float d) {
    float k        = 1.0 / (1.0 + 0.2316419 * abs(d));
    float p        = horner(k);
    float rsqrt2pi = 0.39894228040143267793994605993438;
    return rsqrt2pi * exp(-0.5*d*d) * p;
  }

  __host__ __device__ static
  float cnd(float d) {
    float c = cnd0(d);
    return 0.0 < d ? 1.0 - c : c;
  }

  int i;
  const double *d_rs;
  const double *d_vs;
  const int days;

  typedef make_option self_type;

public:
  __host__ __device__
  make_option(int i, const double *d_rs, const double *d_vs, int days) :
    i(i), d_rs(d_rs), d_vs(d_vs), days(days) {}

  typedef std::random_access_iterator_tag iterator_category;
  typedef double value_type;
  typedef int difference_type;
  typedef double* pointer;
  typedef double reference;

  __host__ __device__
  double value_at(int i) const {
    int option = i / days;
    int day = i % days;
    double r = d_rs[option];
    double v = d_vs[option];

    bool call = day % 2 == 0;

    double price = 58 + 5 * (1+day)/double(days);
    double strike = 65;
    double years = (1+day)/365.0;
    double v_sqrtT = v * sqrt(years);
    double d1      = (log(price / strike) + (r + 0.5 * v * v) * years) / v_sqrtT;
    double d2      = d1 - v_sqrtT;
    double cndD1   = cnd(d1);
    double cndD2   = cnd(d2);
    double x_expRT = strike * exp(-r * years);

    if (call) {
      return price * cndD1 - x_expRT * cndD2;
    } else {
      return x_expRT * (1.0 - cndD2) - price * (1.0 - cndD1);
    }
  }


  __device__
  double operator*() const {
    return value_at(i);
  }

  __host__ __device__ self_type operator++(int)
  {
    self_type retval = *this;
    i++;
    return retval;
  }

  __host__ __device__ __forceinline__ self_type operator++()
  {
    i++;
    return *this;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator+(Distance n) const
  {
    self_type retval(i + int(n), d_rs, d_vs, days);
    return retval;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
  {
    i += (int) n;
    return *this;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type operator-(Distance n) const
  {
    self_type retval(i - (int)n, d_rs, d_vs, days);
    return retval;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
  {
    i -= n;
    return *this;
  }

  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance n) const
  {
    return value_at(i+(int)n);
  }
};

int main(int argc, char** argv) {

  int num_segments = pow(2,atoi(argv[1]));
  int segment_size = pow(2,atoi(argv[2]));

  cerr << num_segments << " segments of " << segment_size << " elements each" << endl;

  int *h_offsets = new int[num_segments+1];

  for (int i = 0; i < num_segments+1; i++) {
    h_offsets[i] = i * segment_size;
  }

  double *h_rs = new double[num_segments];
  double *h_vs = new double[num_segments];

  srand(31337);

  for (int i = 0; i < num_segments; i++) {
    h_rs[i] = rand()/double(RAND_MAX);
    h_vs[i] = rand()/double(RAND_MAX);
  }

  int *d_offsets;
  double *d_rs;
  double *d_vs;
  int *d_out;
  cudaSucceeded(cudaMalloc(&d_offsets, (num_segments+1)*sizeof(int)));
  cudaSucceeded(cudaMalloc(&d_rs, num_segments*sizeof(double)));
  cudaSucceeded(cudaMalloc(&d_vs, num_segments*sizeof(double)));
  cudaSucceeded(cudaMalloc(&d_out, num_segments*sizeof(double)));

  cudaSucceeded(cudaMemcpy(d_offsets, h_offsets, (num_segments+1)*sizeof(int),
                           cudaMemcpyHostToDevice));
  cudaSucceeded(cudaMemcpy(d_rs, h_rs, num_segments*sizeof(double),
                           cudaMemcpyHostToDevice));
  cudaSucceeded(cudaMemcpy(d_vs, h_vs, num_segments*sizeof(double),
                           cudaMemcpyHostToDevice));
  cudaSucceeded(cudaDeviceSynchronize());

  void     *d_temp_storage = NULL;

  // Now time.

  static const int num_runs = 100;
  int total_us = 0;

  // We re-allocate memory for the output and intermediary arrays,
  // because that is also what the Futhark-generated code does
  // (including the computation of how much to allocate).
  for (int i = 0; i < num_runs; i++) {
    cudaSucceeded(cudaFree(d_out));
    cudaSucceeded(cudaFree(d_temp_storage));

    start_timing();
    cudaSucceeded(cudaMalloc(&d_out, num_segments*sizeof(int)));

    size_t temp_storage_bytes = 0;
    cudaSucceeded(cub::DeviceSegmentedReduce::Sum
                  (d_temp_storage, temp_storage_bytes,
                   make_option(0, d_rs, d_vs, segment_size), d_out,
                   num_segments,
                   d_offsets, d_offsets + 1));

    cudaSucceeded(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cudaSucceeded(cub::DeviceSegmentedReduce::Sum
                  (d_temp_storage, temp_storage_bytes,
                   make_option(0, d_rs, d_vs, segment_size), d_out,
                   num_segments,
                   d_offsets, d_offsets + 1));
    end_timing();
    total_us += get_us();
  }
  cerr << total_us/num_runs << "us" << endl;
  if (!isatty(1)) {
    cout << total_us/num_runs;
  }

  int * h_out = new int[num_segments];
  cudaSucceeded(cudaMemcpy(h_out, d_out, num_segments*sizeof(int),
                           cudaMemcpyDeviceToHost));

  // No validation; trust CUB.

  return 0;
}
