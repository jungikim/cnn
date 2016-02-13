#ifndef CNN_EXEC_H
#define CNN_EXEC_H

#include "cnn/cnn.h"

namespace cnn {

class ExecutionEngine {
 public:
  virtual ~ExecutionEngine();
  virtual void invalidate() = 0;
  virtual const Tensor& forward() = 0;
  virtual const Tensor& forward(VariableIndex i) = 0;
  virtual const Tensor& incremental_forward() = 0;  // if you want to add nodes and evaluate just the new parts
  virtual const Tensor& incremental_forward(VariableIndex i) = 0;
  virtual const Tensor& get_value(VariableIndex i) = 0;
  virtual void backward() = 0;
  virtual void backward(VariableIndex i) = 0;
 protected:
  explicit ExecutionEngine(const ComputationGraph& cg) : cg(cg) {}
  const ComputationGraph& cg;
};

class SimpleExecutionEngine : public ExecutionEngine {
 public:
  explicit SimpleExecutionEngine(const ComputationGraph& cg) : ExecutionEngine(cg) {}
  void invalidate();
  const Tensor& forward();
  const Tensor& forward(VariableIndex i);
  const Tensor& incremental_forward();  // if you want to add nodes and evaluate just the new parts
  const Tensor& incremental_forward(VariableIndex i);
  const Tensor& get_value(VariableIndex i);
  void backward();
  void backward(VariableIndex i);
 private:
  std::vector<Tensor> nfxs;
  std::vector<Tensor> ndEdfs;
  VariableIndex num_nodes_evaluated;
};

} // namespace cnn

#endif
