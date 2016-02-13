#ifndef CNN_CONV_H_
#define CNN_CONV_H_

#include "cnn/cnn.h"

namespace cnn {

struct AddVectorToAllColumns : public Node {
  explicit AddVectorToAllColumns(const std::vector<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const;
  Dim dim_forward(const std::vector<Dim>& xs) const;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const;
};

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::vector<VariableIndex>& a, unsigned k = 1) : Node(a), k(k) {}
  std::string as_string(const std::vector<std::string>& arg_names) const;
  Dim dim_forward(const std::vector<Dim>& xs) const;
  size_t aux_storage_size() const;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const;
  unsigned k;
};

struct FoldRows : public Node {
  explicit FoldRows(const std::vector<VariableIndex>& a, unsigned nrows) : Node(a), nrows(nrows) {}
  std::string as_string(const std::vector<std::string>& arg_names) const;
  Dim dim_forward(const std::vector<Dim>& xs) const;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const;
  unsigned nrows;
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DNarrow : public Node {
  explicit Conv1DNarrow(const std::vector<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const;
  Dim dim_forward(const std::vector<Dim>& xs) const;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const;
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DWide : public Node {
  explicit Conv1DWide(const std::vector<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const;
  Dim dim_forward(const std::vector<Dim>& xs) const;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const;
};

} // namespace cnn

#endif
