#include "cnn/expr.h"

#include <initializer_list>

#include "cnn/nodes.h"
#include "cnn/conv.h"

namespace cnn { namespace expr {

using std::vector;

Expression input(ComputationGraph& g, real s) { return Expression(&g, g.add_input(s)); }
Expression input(ComputationGraph& g, const real *ps) { return Expression(&g, g.add_input(ps)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>& data) { return Expression(&g, g.add_input(d, data)); }
Expression input(ComputationGraph& g, const Dim& d, const vector<float>* pdata) { return Expression(&g, g.add_input(d, pdata)); }
Expression const_parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_const_parameters(p)); }
Expression parameter(ComputationGraph& g, Parameters* p) { return Expression(&g, g.add_parameters(p)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, unsigned index) { return Expression(&g, g.add_lookup(p, index)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex) { return Expression(&g, g.add_lookup(p, pindex)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const vector<unsigned>& indices) { return Expression(&g, g.add_lookup(p, indices)); }
Expression lookup(ComputationGraph& g, LookupParameters* p, const vector<unsigned>* pindices) { return Expression(&g, g.add_lookup(p, pindices)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, unsigned index) { return Expression(&g, g.add_const_lookup(p, index)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, const unsigned* pindex) { return Expression(&g, g.add_const_lookup(p, pindex)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, const vector<unsigned>& indices) { return Expression(&g, g.add_const_lookup(p, indices)); }
Expression const_lookup(ComputationGraph& g, LookupParameters* p, const vector<unsigned>* pindices) { return Expression(&g, g.add_const_lookup(p, pindices)); }
Expression zeroes(ComputationGraph& g, const Dim& d) { return Expression(&g, g.add_function<Zeroes>(d)); }

Expression operator-(const Expression& x) { return Expression(x.pg, x.pg->add_function<Negate>(vector_of<VariableIndex>(x.i))); }
Expression operator+(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Sum>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression operator+(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantPlusX>(vector_of<VariableIndex>(y.i), x)); }
Expression operator+(const Expression& x, real y) { return y+x; }
Expression operator-(const Expression& x, const Expression& y) { return x+(-y); }
Expression operator-(real x, const Expression& y) { return Expression(y.pg, y.pg->add_function<ConstantMinusX>(vector_of<VariableIndex>(y.i), x)); }
Expression operator-(const Expression& x, real y) { return -(y-x); }
Expression operator*(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<MatrixMultiply>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression operator*(const Expression& x, float y) { return Expression(x.pg, x.pg->add_function<ConstScalarMultiply>(vector_of<VariableIndex>(x.i), y)); }
Expression addmv(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<AddMv>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression cdiv(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<CwiseQuotient>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression colwise_add(const Expression& x, const Expression& bias) { return Expression(x.pg, x.pg->add_function<AddVectorToAllColumns>(vector_of<VariableIndex>(x.i)(bias.i))); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>(vector_of<VariableIndex>(x.i)(y.i)(z.i))); }
Expression contract3d_1d_1d(const Expression& x, const Expression& y, const Expression& z, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D_1D>(vector_of<VariableIndex>(x.i)(y.i)(z.i)(b.i))); }
Expression contract3d_1d(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression contract3d_1d(const Expression& x, const Expression& y, const Expression& b) { return Expression(x.pg, x.pg->add_function<InnerProduct3D_1D>(vector_of<VariableIndex>(x.i)(y.i)(b.i))); }

Expression sqrt(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sqrt>(vector_of<VariableIndex>(x.i))); }
Expression erf(const Expression& x) { return Expression(x.pg, x.pg->add_function<Erf>(vector_of<VariableIndex>(x.i))); }
Expression tanh(const Expression& x) { return Expression(x.pg, x.pg->add_function<Tanh>(vector_of<VariableIndex>(x.i))); }
Expression lgamma(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogGamma>(vector_of<VariableIndex>(x.i))); }
Expression log(const Expression& x) { return Expression(x.pg, x.pg->add_function<Log>(vector_of<VariableIndex>(x.i))); }
Expression exp(const Expression& x) { return Expression(x.pg, x.pg->add_function<Exp>(vector_of<VariableIndex>(x.i))); }
Expression square(const Expression& x) { return Expression(x.pg, x.pg->add_function<Square>(vector_of<VariableIndex>(x.i))); }
Expression cube(const Expression& x) { return Expression(x.pg, x.pg->add_function<Cube>(vector_of<VariableIndex>(x.i))); }
Expression logistic(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogisticSigmoid>(vector_of<VariableIndex>(x.i))); }
Expression rectify(const Expression& x) { return Expression(x.pg, x.pg->add_function<Rectify>(vector_of<VariableIndex>(x.i))); }
Expression hinge(const Expression& x, unsigned index, float m) { return Expression(x.pg, x.pg->add_function<Hinge>(vector_of<VariableIndex>(x.i), index, m)); }
Expression hinge(const Expression& x, const unsigned* pindex, float m) { return Expression(x.pg, x.pg->add_function<Hinge>(vector_of<VariableIndex>(x.i), pindex, m)); }
Expression log_softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogSoftmax>(vector_of<VariableIndex>(x.i))); }
Expression log_softmax(const Expression& x, const vector<unsigned>& d) { return Expression(x.pg, x.pg->add_function<RestrictedLogSoftmax>(vector_of<VariableIndex>(x.i), d)); }
Expression sparsemax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Sparsemax>(vector_of<VariableIndex>(x.i))); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>& target_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>(vector_of<VariableIndex>(x.i), target_support)); }
Expression sparsemax_loss(const Expression& x, const vector<unsigned>* ptarget_support) { return Expression(x.pg, x.pg->add_function<SparsemaxLoss>(vector_of<VariableIndex>(x.i), ptarget_support)); }
Expression softmax(const Expression& x) { return Expression(x.pg, x.pg->add_function<Softmax>(vector_of<VariableIndex>(x.i))); }
Expression softsign(const Expression& x) { return Expression(x.pg, x.pg->add_function<SoftSign>(vector_of<VariableIndex>(x.i))); }
Expression pow(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Pow>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression min(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Min>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression max(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<Max>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression noise(const Expression& x, real stddev) { return Expression(x.pg, x.pg->add_function<GaussianNoise>(vector_of<VariableIndex>(x.i), stddev)); }
Expression dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<Dropout>(vector_of<VariableIndex>(x.i), p)); }
Expression block_dropout(const Expression& x, real p) { return Expression(x.pg, x.pg->add_function<BlockDropout>(vector_of<VariableIndex>(x.i), p)); }

Expression reshape(const Expression& x, const Dim& d) { return Expression(x.pg, x.pg->add_function<Reshape>(vector_of<VariableIndex>(x.i), d)); }
Expression transpose(const Expression& x) { return Expression(x.pg, x.pg->add_function<Transpose>(vector_of<VariableIndex>(x.i))); }
Expression select_rows(const Expression& x, const vector<unsigned>& rows) { return Expression(x.pg, x.pg->add_function<SelectRows>(vector_of<VariableIndex>(x.i), rows)); }
Expression select_rows(const Expression& x, const vector<unsigned>* prows) { return Expression(x.pg, x.pg->add_function<SelectRows>(vector_of<VariableIndex>(x.i), prows)); }
Expression select_cols(const Expression& x, const vector<unsigned>& cols) { return Expression(x.pg, x.pg->add_function<SelectCols>(vector_of<VariableIndex>(x.i), cols)); }
Expression select_cols(const Expression& x, const vector<unsigned>* pcols) { return Expression(x.pg, x.pg->add_function<SelectCols>(vector_of<VariableIndex>(x.i), pcols)); }
Expression inverse(const Expression& x) { return Expression(x.pg, x.pg->add_function<MatrixInverse>(vector_of<VariableIndex>(x.i))); }
Expression logdet(const Expression& x) { return Expression(x.pg, x.pg->add_function<LogDet>(vector_of<VariableIndex>(x.i))); }

Expression trace_of_product(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<TraceOfProduct>(vector_of<VariableIndex>(x.i)(y.i)));}
Expression cwise_multiply(const Expression& x, const Expression& y) {return Expression(x.pg, x.pg->add_function<CwiseMultiply>(vector_of<VariableIndex>(x.i)(y.i)));}

Expression squared_norm(const Expression& x) { return Expression(x.pg, x.pg->add_function<SquaredNorm>(vector_of<VariableIndex>(x.i))); }

Expression dot_product(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<DotProduct>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression squared_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<SquaredEuclideanDistance>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression huber_distance(const Expression& x, const Expression& y, real c) { return Expression(x.pg, x.pg->add_function<HuberDistance>(vector_of<VariableIndex>(x.i)(y.i), c)); }
Expression l1_distance(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<L1Distance>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression binary_log_loss(const Expression& x, const Expression& y) { return Expression(x.pg, x.pg->add_function<BinaryLogLoss>(vector_of<VariableIndex>(x.i)(y.i))); }
Expression pairwise_rank_loss(const Expression& x, const Expression& y, real m) { return Expression(x.pg, x.pg->add_function<PairwiseRankLoss>(vector_of<VariableIndex>(x.i)(y.i), m)); }
Expression poisson_loss(const Expression& x, unsigned y) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>(vector_of<VariableIndex>(x.i), y)); }
Expression poisson_loss(const Expression& x, const unsigned* py) { return Expression(x.pg, x.pg->add_function<PoissonRegressionLoss>(vector_of<VariableIndex>(x.i), py)); }

Expression conv1d_narrow(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DNarrow>(vector_of<VariableIndex>(x.i)(f.i))); }
Expression conv1d_wide(const Expression& x, const Expression& f) { return Expression(x.pg, x.pg->add_function<Conv1DWide>(vector_of<VariableIndex>(x.i)(f.i))); }
Expression kmax_pooling(const Expression& x, unsigned k) { return Expression(x.pg, x.pg->add_function<KMaxPooling>(vector_of<VariableIndex>(x.i), k)); }
Expression fold_rows(const Expression& x, unsigned nrows) { return Expression(x.pg, x.pg->add_function<FoldRows>(vector_of<VariableIndex>(x.i), nrows)); }

Expression pick(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickElement>(vector_of<VariableIndex>(x.i), v)); }
Expression pick(const Expression& x, const vector<unsigned> & v) { return Expression(x.pg, x.pg->add_function<PickElement>(vector_of<VariableIndex>(x.i), v)); }
Expression pick(const Expression& x, unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickElement>(vector_of<VariableIndex>(x.i), pv)); }
Expression pick(const Expression& x, const vector<unsigned> * pv) { return Expression(x.pg, x.pg->add_function<PickElement>(vector_of<VariableIndex>(x.i), pv)); }

Expression pickrange(const Expression& x, unsigned v, unsigned u) { return Expression(x.pg, x.pg->add_function<PickRange>(vector_of<VariableIndex>(x.i), v, u)); }

Expression pickneglogsoftmax(const Expression& x, unsigned v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>(vector_of<VariableIndex>(x.i), v)); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> & v) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>(vector_of<VariableIndex>(x.i), v)); }
Expression pickneglogsoftmax(const Expression& x, unsigned* pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>(vector_of<VariableIndex>(x.i), pv)); }
Expression pickneglogsoftmax(const Expression& x, const vector<unsigned> * pv) { return Expression(x.pg, x.pg->add_function<PickNegLogSoftmax>(vector_of<VariableIndex>(x.i), pv)); }

Expression sum_cols(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumColumns>(vector_of<VariableIndex>(x.i))); }

Expression sum_batches(const Expression& x) { return Expression(x.pg, x.pg->add_function<SumBatches>(vector_of<VariableIndex>(x.i))); }

Expression kmh_ngram(const Expression& x, unsigned n) { return Expression(x.pg, x.pg->add_function<KMHNGram>(vector_of<VariableIndex>(x.i), n)); }

} }
