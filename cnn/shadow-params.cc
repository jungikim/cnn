#include "cnn/cnn.h"
#include "cnn/shadow-params.h"
#include "cnn/tensor.h"
#include "cnn/aligned-mem-pool.h"
#include "cnn/model.h"

#include "boost/foreach.hpp"

using namespace std;

namespace cnn {

ShadowParameters::ShadowParameters(const Parameters& p) : h(p.values) {
  h.v = (float*)default_device->mem->malloc(h.d.size() * sizeof(float));
  TensorTools::Zero(h);
}

ShadowLookupParameters::ShadowLookupParameters(const LookupParameters& lp) : h(lp.values) {
  BOOST_FOREACH (Tensor& t, h) {
    t.v = (float*)default_device->mem->malloc(t.d.size() * sizeof(float));
    TensorTools::Zero(t);
  }
}

vector<ShadowParameters> AllocateShadowParameters(const Model& m) {
  vector<ShadowParameters> v;
  v.reserve(m.parameters_list().size());
  BOOST_FOREACH (Parameters* p, m.parameters_list())
    v.push_back(ShadowParameters(*p));
  return v;
}

vector<ShadowLookupParameters> AllocateShadowLookupParameters(const Model& m) {
  vector<ShadowLookupParameters> v;
  v.reserve(m.lookup_parameters_list().size());
  BOOST_FOREACH (LookupParameters* p, m.lookup_parameters_list())
    v.push_back(ShadowLookupParameters(*p));
  return v;
}

} // namespace cnn

