#ifndef CNN_GRU_H_
#define CNN_GRU_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"

namespace cnn {

class Model;

struct GRUBuilder : public RNNBuilder {
  GRUBuilder() : RNNBuilder() {}
  explicit GRUBuilder(unsigned layers,
                      unsigned input_dim,
                      unsigned hidden_dim,
                      Model* model);
  Expression back() const { return (cur == -1? h0.back() : h[cur].back()); }
  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const { return final_h(); }
  std::vector<Expression> get_h(RNNPointer i) const { return (i == -1 ? h0 : h[i]); }
  std::vector<Expression> get_s(RNNPointer i) const { return get_h(i); }
  unsigned num_h0_components() const { return layers; }
  void copy(const RNNBuilder & params);

 protected:
  void new_graph_impl(ComputationGraph& cg);
  void start_new_sequence_impl(const std::vector<Expression>& h0);
  Expression add_input_impl(int prev, const Expression& x);

  // first index is layer, then ...
  std::vector<std::vector<Parameters*> > params;

  // first index is layer, then ...
  std::vector<std::vector<Expression> > param_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression> > h;

  // initial values of h at each layer
  // - default to zero matrix input
  std::vector<Expression> h0;

  unsigned hidden_dim;
  unsigned layers;
};

} // namespace cnn

#endif
