/*trees_layer.h*/

#ifndef TREES_HELPERS_H_
#define TREES_HELPERS_H_

#include <torch/extension.h>

namespace torch {

// using Matrix =
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// using ConstMatrixMap = Eigen::Map<const Matrix>;
// using MatrixMap = Eigen::Map<Matrix>;

// // NTUtils contains helper functions for type conversion.
// struct NTUtils {
//   static ConstMatrixMap TensorToEigenMatrixReadOnly(const Tensor *tensor,
//                                                     const int num_rows,
//                                                     const int num_cols) {
//     return ConstMatrixMap(tensor->flat<float>().data(), num_rows, num_cols);
//   }

//   static MatrixMap TensorToEigenMatrix(Tensor *tensor, const int num_rows,
//                                        const int num_cols) {
//     return MatrixMap(tensor->flat<float>().data(), num_rows, num_cols);
//   }
// };

// Stores a node in the binary decision tree.
struct Node {
  float root_to_node_prob;
  float weight_input_dot_product;
  float routing_left_prob;
  bool reachable_descendant_leaf = false;
  double sum_g = 0;
};

// A smooth approximation to the indicator function.
// smooth_step_param must be >= 0.
float SmoothIndicator(const float v, const float smooth_step_param);

// Derivative w.r.t. to SmoothIndicator's input.
// smooth_step_param must be >= 0.
float SmoothIndicatorDerivative(const float v, const float smooth_step_param);

// Performs a forward pass over the tree while identifying reachable leaves.
// Returns (i) the output vector, (ii) the updated tree, and (iii) a
// vector of reachable leaves (contains the indices of the reachable leaves in
// tree_nodes).
void ForwardPassSingleSample(const torch::Tensor &node_weights,
                             const torch::Tensor &leaf_weights,
                             const torch::Tensor &input_features,
                             const int depth, const float smooth_step_param,
                             const bool training_mode, torch::Tensor *output,
                             std::vector<Node> *tree_nodes,
                             std::vector<int> *reachable_leaves);

// Returns the gradients w.r.t. to the inputs of the tree, internal nodes, and
// leaves. Internally calls ForwardPassSingleSample to efficiently construct the
// tree.
void BackwardPassSingleSample(const torch::Tensor &node_weights,
                              const torch::Tensor &leaf_weights,
                              const torch::Tensor &input_features,
                              const torch::Tensor &grad_loss_wrt_tree_output,
                              const int depth, const float smooth_step_param,
                              torch::Tensor *grad_loss_wrt_input_features,
                              torch::Tensor *grad_loss_wrt_node_weights,
                              torch::Tensor *grad_loss_wrt_leaf_weights);

} // namespace torch

#endif // TREES_HELPERS_H_
