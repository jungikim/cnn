#include "cnn/tensor.h"

#include "cnn/random.h"

#include <vector>
#include <cstring>

#if HAVE_CUDA
#include "cnn/cuda.h"
#endif

using namespace std;

namespace cnn {

ostream& operator<<(ostream& os, const Tensor& t) {
#if HAVE_CUDA
  vector<real> vt = as_vector(t);
  Eigen::Map<Eigen::MatrixXf> m(&vt[0], t.d.rows(), t.d.cols());
  os << m;
#else
  os << (*t);
#endif
  return os;
}

real as_scalar(const Tensor& t) {
  assert(t.d.size() == 1);
#if HAVE_CUDA
  float res;
  CUDA_CHECK(cudaMemcpy(&res, t.v, sizeof(float), cudaMemcpyDeviceToHost));
  return res;
#else
  return t.v[0];
#endif
}

vector<real> as_vector(const Tensor& v) {
  vector<real> res(v.d.size());
#if HAVE_CUDA
  CUDA_CHECK(cudaMemcpy(&res[0], v.v, sizeof(real) * res.size(), cudaMemcpyDeviceToHost));
#else
  memcpy(&res[0], v.v, sizeof(real) * res.size());
#endif
  return res;
}

float TensorTools::AccessElement(const Tensor& v, int index) {
#if HAVE_CUDA
  float ret;
  cudaMemcpyAsync(&ret, &v.v[index], sizeof(real), cudaMemcpyDeviceToHost);
  return ret;
#else
  return v.v[index];
#endif
}

float TensorTools::AccessElement(const Tensor& v, const Dim& index) {
#if HAVE_CUDA
  abort();
#else
  return (*v)(index[0], index[1]);
#endif
}

void TensorTools::SetElement(const Tensor& v, int index, float value) {
#if HAVE_CUDA
  cudaMemcpyAsync(&v.v[index], &value, sizeof(real), cudaMemcpyHostToDevice);
#else
  v.v[index] = value;
#endif
}

void TensorTools::SetElements(const Tensor& v, const vector<float>& vec) {
#if HAVE_CUDA
  cudaMemcpyAsync(v.v, &vec[0], sizeof(real) * vec.size(), cudaMemcpyHostToDevice);
#else
  memcpy(v.v, &vec[0], sizeof(real) * vec.size());
#endif
}

void TensorTools::CopyElements(const Tensor& v, const Tensor& v_src) {
#if HAVE_CUDA
  cudaMemcpyAsync(v.v, v_src.v, sizeof(real) * v.d.size(), cudaMemcpyDeviceToDevice);
#else
  memcpy(v.v, v_src.v, sizeof(real) * v.d.size());
#endif
}

void TensorTools::Constant(Tensor& d, float c) {
#if HAVE_CUDA
  if (!c) {
    CUDA_CHECK(cudaMemsetAsync(d.v, 0, d.d.size() * sizeof(float)));
  } else {
    fill(d.v, d.v + d.d.size(), c);
  }
#else
  if (!c) {
    memset(d.v, c, d.d.size() * sizeof(float));
  } else {
    fill(d.v, d.v + d.d.size(), c);
  }
#endif
}

void TensorTools::Zero(Tensor& d) {
  Constant(d, 0);
}

void TensorTools::Randomize(Tensor& val, real scale) {
  boost::random::uniform_real_distribution<real> distribution(-scale,scale);
#if HAVE_CUDA
  float* t = new float[val.d.size()];
  for(size_t i = 0 ; i < val.d.size() ; i ++) t[i] = distribution(*rndeng);
  CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
  delete[] t;
#else
  for(size_t i = 0 ; i < val.d.size() ; i ++) val.v[i] = distribution(*rndeng);
#endif
}

void TensorTools::Randomize(Tensor& d) {
  Randomize(d, sqrt(6) / sqrt(d.d.sum_dims()));
}

void TensorTools::RandomBernoulli(Tensor& val, real p, real scale) {
  boost::random::bernoulli_distribution<real> distribution(p);
#if HAVE_CUDA
  float* t = new float[val.d.size()];
  for(size_t i = 0 ; i < val.d.size() ; i ++) t[i] = distribution(*rndeng) * scale;
  CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
  delete[] t;
#else
  for(size_t i = 0 ; i < val.d.size() ; i ++) val.v[i] = distribution(*rndeng) * scale;
#endif
}

void TensorTools::RandomizeNormal(real mean, real stddev, Tensor& val) {
  boost::random::normal_distribution<real> distribution(mean, stddev);
#if HAVE_CUDA
  float* t = new float[val.d.size()];
  for(size_t i = 0 ; i < val.d.size() ; i ++) t[i] = distribution(*rndeng);
  CUDA_CHECK(cudaMemcpy(val.v, t, sizeof(real) * val.d.size(), cudaMemcpyHostToDevice));
  delete[] t;
#else
  for(size_t i = 0 ; i < val.d.size() ; i ++) val.v[i] = distribution(*rndeng);
#endif
}

real rand01() {
  boost::random::uniform_real_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

int rand0n(int n) {
  assert(n > 0);
  int x = rand01() * n;
  while(n == x) { x = rand01() * n; }
  return x;
}

real rand_normal() {
  boost::random::normal_distribution<real> distribution(0, 1);
  return distribution(*rndeng);
}

} // namespace cnn
