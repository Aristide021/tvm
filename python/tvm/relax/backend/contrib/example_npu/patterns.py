# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Example NPU Pattern Table

This module demonstrates how to define operator patterns for a custom NPU backend.
It shows a minimal yet complete pattern registration for common neural network operations.
"""

from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import PatternCheckContext

from ...pattern_registry import register_patterns


def _check_default(context: PatternCheckContext) -> bool:
    """
    Default pattern checker that always returns True.
    
    In a real implementation, this would check hardware constraints like:
    - Supported data types (int8, float16, float32, etc.)
    - Tensor shapes and dimensions
    - Memory limitations
    - Operator-specific constraints
    """
    return True


def linear_patterns():
    """
    Define patterns for linear/dense operations.
    
    This demonstrates matching matrix multiplication operations
    that can be offloaded to NPU hardware accelerators.
    """
    
    def _make_linear_pattern():
        input_tensor = wildcard()
        weight_tensor = wildcard()
        output = is_op("relax.matmul")(input_tensor, weight_tensor)
        annotations = {
            "input": input_tensor,
            "weight": weight_tensor,
            "root": output
        }
        return output, annotations
    
    def _linear_pattern(pattern_name):
        return (pattern_name, *_make_linear_pattern(), _check_default)
    
    return [_linear_pattern("example_npu.dense")]


def conv1d_patterns():
    """
    Define patterns for 1D convolution operations.
    
    Shows how to match convolution operations for NPU offloading.
    Many NPUs have specialized convolution engines.
    """
    
    def _make_conv1d_pattern():
        input_tensor = wildcard()
        weight_tensor = wildcard()
        output = is_op("relax.nn.conv1d")(input_tensor, weight_tensor)
        annotations = {
            "input": input_tensor,
            "weight": weight_tensor,
            "root": output
        }
        return output, annotations
    
    def _conv1d_pattern(pattern_name):
        return (pattern_name, *_make_conv1d_pattern(), _check_default)
    
    return [_conv1d_pattern("example_npu.conv1d")]


def conv2d_patterns():
    """
    Define patterns for 2D convolution operations.
    
    2D convolutions are the most common operation in CNNs
    and are typically well-supported by NPU hardware.
    """
    
    def _make_conv2d_pattern():
        input_tensor = wildcard()
        weight_tensor = wildcard()
        output = is_op("relax.nn.conv2d")(input_tensor, weight_tensor)
        annotations = {
            "input": input_tensor,
            "weight": weight_tensor,
            "root": output
        }
        return output, annotations
    
    def _conv2d_pattern(pattern_name):
        return (pattern_name, *_make_conv2d_pattern(), _check_default)
    
    return [_conv2d_pattern("example_npu.conv2d")]


def activation_patterns():
    """
    Define patterns for activation functions.
    
    Common activation functions that NPUs typically support
    with dedicated hardware units.
    """
    
    def _make_relu_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.relu")(input_tensor)
        annotations = {"input": input_tensor, "root": output}
        return output, annotations
    
    def _make_sigmoid_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.sigmoid")(input_tensor)
        annotations = {"input": input_tensor, "root": output}
        return output, annotations
    
    def _activation_pattern(pattern_name, make_pattern):
        pattern, annotations = make_pattern()
        return (pattern_name, pattern, annotations, _check_default)
    
    return [
        _activation_pattern("example_npu.relu", _make_relu_pattern),
        _activation_pattern("example_npu.sigmoid", _make_sigmoid_pattern),
    ]


def pooling_patterns():
    """
    Define patterns for pooling operations.
    
    Pooling operations are commonly accelerated in NPUs
    as they're essential for CNNs.
    """
    
    def _make_maxpool2d_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.max_pool2d")(input_tensor)
        annotations = {"input": input_tensor, "root": output}
        return output, annotations
    
    def _pooling_pattern(pattern_name):
        pattern, annotations = _make_maxpool2d_pattern()
        return (pattern_name, pattern, annotations, _check_default)
    
    return [_pooling_pattern("example_npu.max_pool2d")]


# Register all patterns with the pattern registry
# This makes them available for graph partitioning
register_patterns(
    [
        *linear_patterns(),
        *conv1d_patterns(),
        *conv2d_patterns(),
        *activation_patterns(),
        *pooling_patterns(),
    ]
)