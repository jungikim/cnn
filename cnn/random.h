#ifndef CNN_EIGEN_RANDOM_H
#define CNN_EIGEN_RANDOM_H

#include "boost/random/random_device.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_real_distribution.hpp"
#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_int_distribution.hpp"
#include "boost/random/bernoulli_distribution.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/variate_generator.hpp"
#include "boost/random/uniform_int.hpp"

namespace cnn {

extern boost::random::mt19937* rndeng;

} // namespace cnn

#endif
