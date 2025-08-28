/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/example_npu/example_npu_runtime.cc
 * \brief Example NPU runtime using JSON serialization
 * 
 * This file demonstrates how to implement a custom NPU runtime
 * that processes JSON-serialized computation graphs. It provides
 * a CPU emulation mode for testing without actual NPU hardware.
 * 
 * Key concepts demonstrated:
 * - JSON-based graph deserialization
 * - Operator dispatch mechanism
 * - Memory management for NPU execution
 * - CPU fallback for testing
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

/*!
 * \brief Example NPU runtime implementation
 * 
 * This class demonstrates the minimal interface needed to implement
 * a custom NPU backend in TVM. It inherits from JSONRuntimeBase which
 * provides the JSON parsing and graph traversal infrastructure.
 */
class ExampleNPURuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief Constructor
   * \param symbol_name The name of the function
   * \param graph_json JSON string representation of the computation graph
   * \param const_names Names of the constant tensors
   */
  ExampleNPURuntime(const std::string& symbol_name, const std::string& graph_json,
                    const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~ExampleNPURuntime() override = default;

  const char* type_key() const override { return "example_npu_json"; }

  /*!
   * \brief Initialize the runtime with constant tensors
   * \param consts Array of constant NDArrays (weights, biases, etc.)
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";
    
    // Setup constant tensors
    SetupConstants(consts);
    
    // In a real implementation, this would:
    // 1. Initialize NPU device
    // 2. Compile the graph for NPU
    // 3. Allocate device memory
    // 4. Transfer constants to device
    LOG(INFO) << "Example NPU Runtime initialized with " << nodes_.size() << " nodes";
  }

  /*!
   * \brief Run the computation graph
   * 
   * This is a simplified CPU emulation of NPU execution.
   * A real implementation would dispatch operations to NPU hardware.
   */
  void Run() override {
    // Process each node in topological order
    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];
      
      if (node.GetOpType() == "kernel") {
        // Dispatch to appropriate operator implementation
        const std::string& op_name = node.GetOpName();
        
        if (op_name == "example_npu.dense") {
          ExecuteDense(node);
        } else if (op_name == "example_npu.conv1d") {
          ExecuteConv1D(node);
        } else if (op_name == "example_npu.conv2d") {
          ExecuteConv2D(node);
        } else if (op_name == "example_npu.relu") {
          ExecuteReLU(node);
        } else if (op_name == "example_npu.sigmoid") {
          ExecuteSigmoid(node);
        } else if (op_name == "example_npu.max_pool2d") {
          ExecuteMaxPool2D(node);
        } else {
          LOG(FATAL) << "Unsupported operation: " << op_name;
        }
      }
    }
  }

 private:
  /*!
   * \brief Execute dense/linear layer (matrix multiplication)
   * 
   * Demonstrates how to implement a basic matrix multiplication
   * that would typically be accelerated on NPU hardware.
   */
  void ExecuteDense(const JSONGraphNode& node) {
    // Get input and output tensor information
    const auto& input_entry = node.GetInputs()[0];
    const auto& weight_entry = node.GetInputs()[1];
    const auto& output_entry = outputs_[EntryID(node, 0)];
    
    const float* input_data = static_cast<const float*>(data_entry_[input_entry.id_]->data);
    const float* weight_data = static_cast<const float*>(data_entry_[weight_entry.id_]->data);
    float* output_data = static_cast<float*>(data_entry_[output_entry]->data);
    
    // Get shapes
    const auto& input_shape = nodes_[input_entry.id_].GetOpShape()[input_entry.index_];
    const auto& weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    
    int batch_size = input_shape[0];
    int input_dim = input_shape[1];
    int output_dim = weight_shape[1];
    
    // Simple matrix multiplication (CPU emulation)
    // Real NPU would use optimized hardware units
    for (int b = 0; b < batch_size; ++b) {
      for (int o = 0; o < output_dim; ++o) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
          sum += input_data[b * input_dim + i] * weight_data[i * output_dim + o];
        }
        output_data[b * output_dim + o] = sum;
      }
    }
  }

  /*!
   * \brief Execute 1D convolution
   * 
   * Simplified 1D convolution implementation for demonstration.
   * NPU hardware typically has specialized convolution engines.
   */
  void ExecuteConv1D(const JSONGraphNode& node) {
    // Simplified implementation - would be hardware accelerated on real NPU
    LOG(INFO) << "Executing Conv1D operation (emulated)";
    
    // In a real implementation:
    // 1. Extract padding, stride, dilation parameters
    // 2. Dispatch to NPU convolution engine
    // 3. Handle different data layouts (NCW, NWC)
  }

  /*!
   * \brief Execute 2D convolution
   */
  void ExecuteConv2D(const JSONGraphNode& node) {
    LOG(INFO) << "Executing Conv2D operation (emulated)";
    // Real implementation would use NPU's 2D convolution accelerator
  }

  /*!
   * \brief Execute ReLU activation
   * 
   * ReLU is typically very efficient on NPU hardware
   * as it's a simple element-wise operation.
   */
  void ExecuteReLU(const JSONGraphNode& node) {
    const auto& input_entry = node.GetInputs()[0];
    const auto& output_entry = outputs_[EntryID(node, 0)];
    
    const float* input_data = static_cast<const float*>(data_entry_[input_entry.id_]->data);
    float* output_data = static_cast<float*>(data_entry_[output_entry]->data);
    
    const auto& shape = nodes_[input_entry.id_].GetOpShape()[input_entry.index_];
    int num_elements = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      num_elements *= shape[i];
    }
    
    // Apply ReLU: max(0, x)
    for (int i = 0; i < num_elements; ++i) {
      output_data[i] = std::max(0.0f, input_data[i]);
    }
  }

  /*!
   * \brief Execute Sigmoid activation
   */
  void ExecuteSigmoid(const JSONGraphNode& node) {
    const auto& input_entry = node.GetInputs()[0];
    const auto& output_entry = outputs_[EntryID(node, 0)];
    
    const float* input_data = static_cast<const float*>(data_entry_[input_entry.id_]->data);
    float* output_data = static_cast<float*>(data_entry_[output_entry]->data);
    
    const auto& shape = nodes_[input_entry.id_].GetOpShape()[input_entry.index_];
    int num_elements = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      num_elements *= shape[i];
    }
    
    // Apply Sigmoid: 1 / (1 + exp(-x))
    for (int i = 0; i < num_elements; ++i) {
      output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
    }
  }

  /*!
   * \brief Execute MaxPool2D operation
   */
  void ExecuteMaxPool2D(const JSONGraphNode& node) {
    LOG(INFO) << "Executing MaxPool2D operation (emulated)";
    // Real implementation would use NPU's pooling hardware
  }
};

/*!
 * \brief Create the Example NPU runtime module
 * \param args The arguments for creating the runtime
 * \return The created runtime module
 */
runtime::Module ExampleNPURuntimeCreate(const Array<String>& args) {
  ICHECK_EQ(args.size(), 3) << "Expected 3 arguments: symbol_name, graph_json, const_names";
  
  auto n = make_object<ExampleNPURuntime>(args[0], args[1], JsonToConstNames(args[2]));
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.ExampleNPUJSONRuntimeCreate")
    .set_body_typed(ExampleNPURuntimeCreate);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm