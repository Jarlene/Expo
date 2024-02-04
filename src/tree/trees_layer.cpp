/*trees_layer.cpp*/

#include "trees_layer.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/ones.h"
#include "ATen/ops/tensor.h"
#include "pybind11/pybind11.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/enum.h"
#include <math.h>
#include <stack>
#include <vector>

namespace torch {

const float kTolerance = 1e-10;

float SmoothIndicator(const float v, const float smooth_step_param) {
  float out;
  if (v <= -0.5 / smooth_step_param) {
    out = 0;
  } else if (v >= 0.5 / smooth_step_param) {
    out = 1;
  } else {
    const float x = smooth_step_param * v + 0.5;
    const float x_squared = x * x;
    const float x_cubed = x_squared * x;
    out = -2 * x_cubed + 3 * x_squared;
  }
  return out;
}

float SmoothIndicatorDerivative(const float v, const float smooth_step_param) {
  float out;
  if (std::fabs(v) <= 0.5 / smooth_step_param) {
    const float x = smooth_step_param * v + 0.5;
    const float x_squared = x * x;
    out = 6 * smooth_step_param * (-x_squared + x);
  } else {
    out = 0;
  }
  return out;
}

void ForwardPassSingleSample(const torch::Tensor &node_weights,
                             const torch::Tensor &leaf_weights,
                             const torch::Tensor &input_features,
                             const int depth, const float smooth_step_param,
                             const bool training_mode, torch::Tensor *output,
                             std::vector<Node> *tree_nodes,
                             std::vector<int> *reachable_leaves) {
  DCHECK(tree_nodes != nullptr) << "Got a null ptr to tree!";
  // tree allows for more readable indexing (e.g., tree[i]).
  std::vector<Node> &tree = *tree_nodes;
  // Check tree size.
  DCHECK(tree.size() == std::pow(2, depth + 1) - 1)
      << "Inconsistent tree size!";
  // Label of the first leaf (assuming a breadth-first order).
  const int first_leaf_label = (tree.size() + 1) / 2 - 1;
  // Stack of indices (of nodes) to traverse.
  std::stack<int> to_traverse;
  // Initialize root probability.
  tree[0].root_to_node_prob = 1;
  // Push the root index.
  to_traverse.push(0);

  // Fill the tree depth-first, while skipping unreachable nodes.
  // Note: tree is a perfect binary tree.
  while (!to_traverse.empty()) {
    const int current_index = to_traverse.top();
    to_traverse.pop();
    auto node_weights_ind = node_weights.index({"...", current_index});
    tree[current_index].weight_input_dot_product =
        input_features.dot(node_weights_ind).item().toFloat();
    const float probability_left = SmoothIndicator(
        tree[current_index].weight_input_dot_product, smooth_step_param);
    tree[current_index].routing_left_prob = probability_left;
    // Branch left if prob_left is non zero.
    if (tree[current_index].routing_left_prob > kTolerance) {
      const int left_index = 2 * current_index + 1;
      tree[left_index].root_to_node_prob =
          tree[current_index].root_to_node_prob *
          tree[current_index].routing_left_prob;
      // Push to the stack only if left child is an internal node.
      if (left_index < first_leaf_label) {
        to_traverse.push(left_index);
      } else {
        // This is a reachable leaf.
        reachable_leaves->push_back(left_index);
      }
    }

    // Branch right if prob_right is non zero.
    // Note: Not mutually exclusive with the previous if.
    if (1 - tree[current_index].routing_left_prob > kTolerance) {
      const int right_index = 2 * current_index + 2;
      tree[right_index].root_to_node_prob =
          tree[current_index].root_to_node_prob *
          (1 - tree[current_index].routing_left_prob);
      // Push to the stack only if left child is an internal node.
      if (right_index < first_leaf_label) {
        to_traverse.push(right_index);
      } else {
        // This is a reachable leaf.
        reachable_leaves->push_back(right_index);
      }
    }
  }
  output->zero_();

  // Iterate over the (reachable) leaves to update the tree output.
  for (auto i : *reachable_leaves) {
    // (i - first_leaf_label) is the index of the column of leaf i in
    // leaf_weights.
    auto leaf_weights_ind = leaf_weights.index({"...", i - first_leaf_label});
    *output += tree[i].root_to_node_prob * leaf_weights_ind;
  }

  if (training_mode) {
    // Mark all the reachable leaves and their ancestors.
    for (auto i : *reachable_leaves) {
      // Traverse up to the root starting from the current leaf.
      int current_index = i;
      tree[i].reachable_descendant_leaf = true;
      while (current_index != 0) {
        // The body below marks the parent.
        // Is the current_index a left child?
        const bool left_child = (current_index % 2 == 1);
        const int parent_index =
            left_child ? (current_index - 1) / 2 : (current_index - 2) / 2;
        if (tree[parent_index].reachable_descendant_leaf) {
          break;
        } else {
          tree[parent_index].reachable_descendant_leaf = true;
        }
        // Move to the parent.
        current_index = parent_index;
      }
    }
  }
}

void BackwardPassSingleSample(const torch::Tensor &grad_loss_wrt_tree_output,
                              const torch::Tensor &node_weights,
                              const torch::Tensor &leaf_weights,
                              const torch::Tensor &input_features,
                              const int depth, const float smooth_step_param,
                              torch::Tensor *grad_loss_wrt_input_features,
                              torch::Tensor *grad_loss_wrt_node_weights,
                              torch::Tensor *grad_loss_wrt_leaf_weights) {
  const int tree_num_nodes = std::pow(2, depth + 1) - 1;
  std::vector<Node> tree(tree_num_nodes);
  // Label of the first leaf (assuming a breadth-first order).
  const int first_leaf_label = (tree_num_nodes + 1) / 2 - 1;
  const int output_logits_dim = leaf_weights.size(0);
  torch::Tensor output_logits_sample = torch::zeros({output_logits_dim});
  // Eigen::VectorXf output_logits_sample(output_logits_dim);
  std::vector<int> reachable_leaves;
  // Do a forward pass to build the tree and obtain the reachable leaves.
  // TODO: Remove this forward pass and use the results from
  // the previous call to the forward pass.
  ForwardPassSingleSample(node_weights, leaf_weights, input_features, depth,
                          smooth_step_param, true, &output_logits_sample, &tree,
                          &reachable_leaves);

  grad_loss_wrt_input_features->zero_();
  grad_loss_wrt_node_weights->zero_();
  grad_loss_wrt_leaf_weights->zero_();

  // Stacks s1 and s2 are for post order traversal.
  std::stack<int> s1, s2;
  s1.push(0);
  while (!s1.empty()) {
    // Pop an item from s1 and push it to s2.
    const int current_index = s1.top();
    s1.pop();
    s2.push(current_index);
    // Push "reachable" left and right children to s1.
    const int left_index = 2 * current_index + 1;
    if (left_index < tree_num_nodes &&
        tree[left_index].reachable_descendant_leaf) {
      s1.push(left_index);
    }
    const int right_index = left_index + 1;
    if (right_index < tree_num_nodes &&
        tree[right_index].reachable_descendant_leaf) {
      s1.push(right_index);
    }
  }

  // Now do post order traversal by iterating over s2.
  while (!s2.empty()) {
    const int current_index = s2.top();
    s2.pop();
    // Process a leaf.
    if (current_index >= first_leaf_label) {
      // Update grad_loss_wrt_leaf_weights for leaf i.
      // index of current leaf starting from 0.
      const int leaf_zero_based_index = current_index - first_leaf_label;

      grad_loss_wrt_leaf_weights->index_put_(
          {"...", leaf_zero_based_index},
          grad_loss_wrt_tree_output * tree[current_index].root_to_node_prob);

      tree[current_index].sum_g =
          grad_loss_wrt_leaf_weights->index({"...", leaf_zero_based_index})
              .dot(leaf_weights.index({"...", leaf_zero_based_index}))
              .item()
              .toFloat();

    }
    // Process an internal node only if it's fractional (i.e., belongs to the
    // fractional tree).
    else if (tree[current_index].routing_left_prob > 0 &&
             tree[current_index].routing_left_prob < 1) {
      float activation_function_derivative = SmoothIndicatorDerivative(
          tree[current_index].weight_input_dot_product, smooth_step_param);
      // Notation below defined in Algorithm 2 of TEL's paper.
      const double mu_1 = activation_function_derivative /
                          tree[current_index].routing_left_prob;
      const double mu_2 = activation_function_derivative /
                          (1 - tree[current_index].routing_left_prob);
      const int left_index = 2 * current_index + 1;
      const int right_index = left_index + 1;
      const double a_minus_b =
          mu_1 * tree[left_index].sum_g - mu_2 * tree[right_index].sum_g;
      *grad_loss_wrt_input_features +=
          a_minus_b * node_weights.index({"...", current_index});
      grad_loss_wrt_node_weights->index({"...", current_index}) =
          a_minus_b * input_features;
      tree[current_index].sum_g =
          tree[left_index].sum_g + tree[right_index].sum_g;
    } else {
      const int left_index = 2 * current_index + 1;
      const int right_index = left_index + 1;
      tree[current_index].sum_g =
          tree[left_index].sum_g + tree[right_index].sum_g;
    }
  }
}

std::vector<torch::Tensor> ForwardImpl(
    const torch::Tensor &input_features, const torch::Tensor &node_weights,
    const torch::Tensor &leaf_weights, const int output_logits_dim,
    const int depth, const float smooth_step_param, const bool training_mode) {
  const int tree_num_nodes = std::pow(2, depth + 1) - 1;
  float average_num_reachable_leaves = 0;
  int batch_size = input_features.size(0);
  torch::Tensor out = torch::zeros({batch_size, output_logits_dim});
  for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
    std::vector<Node> sample_tree(tree_num_nodes);
    std::vector<int> leaves;
    const auto input_features_sample =
        input_features.index({sample_index, "..."});
    auto output_logits_sample = torch::zeros({output_logits_dim});
    ForwardPassSingleSample(node_weights, leaf_weights, input_features_sample,
                            depth, smooth_step_param, training_mode,
                            &output_logits_sample, &sample_tree, &leaves);
    out.index_put_({sample_index, "..."}, output_logits_sample);
    average_num_reachable_leaves += leaves.size();
  }
  average_num_reachable_leaves /= static_cast<float>(batch_size);
  auto average_num_reachable_leaves_tensor =
      torch::tensor(average_num_reachable_leaves);

  return std::vector<torch::Tensor>{out, average_num_reachable_leaves_tensor};
}

std::vector<torch::Tensor>
BackwardImpl(const torch::Tensor &grad_loss_wrt_tree_output,
             const torch::Tensor &input_features,
             const torch::Tensor &node_weights,
             const torch::Tensor &leaf_weights, const int output_logits_dim,
             const int depth, const float smooth_step_param) {
  const int num_internal_nodes = node_weights.size(1);
  const int input_dim = input_features.size(1);
  const int num_leaves = node_weights.size(1);
  int batch_size = input_features.size(0);
  torch::Tensor grad_loss_wrt_input_features =
      torch::zeros(input_features.sizes());
  torch::Tensor grad_loss_wrt_node_weights =
      torch::zeros({input_dim, num_internal_nodes});
  torch::Tensor grad_loss_wrt_leaf_weights =
      torch::zeros({output_logits_dim, num_internal_nodes});
  for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
    auto grad_loss_wrt_tree_output_sample =
        grad_loss_wrt_tree_output.index({sample_index, "..."});
    auto input_features_sample = input_features.index({sample_index, "..."});

    auto grad_loss_wrt_input_features_sample = torch::zeros({input_dim});
    auto grad_loss_wrt_node_weights_sample =
        torch::zeros({input_dim, num_internal_nodes});
    auto grad_loss_wrt_leaf_weights_sample =
        torch::zeros({output_logits_dim, num_leaves});
    BackwardPassSingleSample(
        grad_loss_wrt_tree_output_sample, node_weights, leaf_weights,
        input_features_sample, depth, smooth_step_param,
        &grad_loss_wrt_input_features_sample,
        &grad_loss_wrt_node_weights_sample, &grad_loss_wrt_leaf_weights_sample);

    grad_loss_wrt_input_features.index_put_(
        {sample_index, "..."}, grad_loss_wrt_input_features_sample);
    grad_loss_wrt_node_weights =
        grad_loss_wrt_node_weights + grad_loss_wrt_node_weights_sample;
    grad_loss_wrt_leaf_weights =
        grad_loss_wrt_leaf_weights + grad_loss_wrt_leaf_weights_sample;
  }

  return std::vector<torch::Tensor>{grad_loss_wrt_input_features,
                                    grad_loss_wrt_node_weights,
                                    grad_loss_wrt_leaf_weights};
}

class TreeFunc : autograd::Function<TreeFunc> {
public:
  static autograd::variable_list
  forward(torch::autograd::AutogradContext *ctx,
          autograd::Variable &input_features, autograd::Variable &node_weights,
          autograd::Variable &leaf_weights, int output_logits_dim, int depth,
          float smooth_step_param, bool training_mode) {

    ctx->saved_data["output_logits_dim"] = output_logits_dim;
    ctx->saved_data["depth"] = depth;
    ctx->saved_data["smooth_step_param"] = smooth_step_param;
    ctx->save_for_backward({input_features, node_weights, leaf_weights});
    return ForwardImpl(node_weights, leaf_weights, input_features,
                       output_logits_dim, depth, smooth_step_param,
                       training_mode);
  };

  static autograd::variable_list backward(autograd::AutogradContext *ctx,
                                          autograd::variable_list grad_output) {
    auto output_logits_dim = ctx->saved_data["output_logits_dim"].toInt();
    auto depth = ctx->saved_data["depth"].toInt();
    auto smooth_step_param = ctx->saved_data["smooth_step_param"].toDouble();
    auto parmas = ctx->get_saved_variables();
    return BackwardImpl(grad_output[0], parmas[0], parmas[1], parmas[2],
                        output_logits_dim, depth, smooth_step_param);
  };

  template <typename... Args> static auto applies(Args &&...args) {
    return autograd::Function<TreeFunc>::apply<TreeFunc, Args...>(
        std::forward<Args>(args)...);
  };
};

} // namespace torch

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // py::class_<torch::TreeFunc>(m, "TreeFunc")
  //     .def(py::init<>())
  //     .def("apply", &torch::TreeFunc::applies<
  //                       torch::autograd::Variable, torch::autograd::Variable,
  //                       torch::autograd::Variable, int, int, float, bool>);
  m.def("forward", &torch::ForwardImpl);
  m.def("backward", &torch::BackwardImpl);
}