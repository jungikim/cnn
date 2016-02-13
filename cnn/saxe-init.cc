#include "cnn/saxe-init.h"
#include "cnn/tensor.h"
#include "cnn/random.h"

#include <cstring>

#include <Eigen/SVD>

using namespace std;

namespace cnn {

void OrthonormalRandom(unsigned dd, float g, Tensor& x) {
  Tensor t;
  t.d = Dim(vector_of<unsigned int>(dd)(dd));
  t.v = new float[dd * dd];
  boost::random::normal_distribution<float> distribution(0, 0.01);
  for(size_t i = 0 ; i < dd*dd ; i ++) t.v[i] = distribution(*rndeng);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(*t, Eigen::ComputeFullU);
  *x = svd.matrixU();
  delete[] t.v;
}

}

