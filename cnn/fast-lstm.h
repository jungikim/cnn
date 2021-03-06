#ifndef CNN_FAST_LSTM_H_
#define CNN_FAST_LSTM_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"

#include "boost/foreach.hpp"

using namespace cnn::expr;

namespace cnn {

class Model;

/*
FastLSTM replaces the matrices from cell to other units, by diagonal matrices.
*/
struct FastLSTMBuilder : public RNNBuilder {
  FastLSTMBuilder() : RNNBuilder() {}
  explicit FastLSTMBuilder(unsigned layers,
                           unsigned input_dim,
                           unsigned hidden_dim,
                           Model* model);

  Expression back() const { return (cur == -1? h0.back() : h[cur].back()); }
  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const {
    std::vector<Expression> ret = (c.size() == 0 ? c0 : c.back());
    BOOST_FOREACH (Expression my_h, final_h()) ret.push_back(my_h);
    return ret;
  }
  unsigned num_h0_components() const { return 2 * layers; }

  std::vector<Expression> get_h(RNNPointer i) const { return (i == -1 ? h0 : h[i]); }
  std::vector<Expression> get_s(RNNPointer i) const {
    std::vector<Expression> ret = (i == -1 ? c0 : c[i]);
    BOOST_FOREACH (Expression my_h, get_h(i)) ret.push_back(my_h);
    return ret;
  }

  void copy(const RNNBuilder & params);
 protected:
  void new_graph_impl(ComputationGraph& cg);
  void start_new_sequence_impl(const std::vector<Expression>& h0);
  Expression add_input_impl(int prev, const Expression& x);

 public:
  // first index is layer, then ...
  std::vector<std::vector<Parameters*> > params;

  // first index is layer, then ...
  std::vector<std::vector<Expression> > param_vars;

  // first index is time, second is layer
  std::vector<std::vector<Expression> > h, c;

  // initial values of h and c at each layer
  // - both default to zero matrix input
  bool has_initial_state; // if this is false, treat h0 and c0 as 0
  std::vector<Expression> h0;
  std::vector<Expression> c0;
  unsigned layers;
};

} // namespace cnn

#endif
