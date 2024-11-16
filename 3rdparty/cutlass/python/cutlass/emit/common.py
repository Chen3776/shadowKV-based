#################################################################################################
#
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Common utilities for emitting CUTLASS kernels
"""

import cutlass

# Strings used for printing information about the generation of emitted scripts
_AUTOGEN_STR = f"This file was automatically generated by the CUTLASS {cutlass.__version__} Python interface (https://github.com/nvidia/cutlass/python)"


_CSTYLE_AUTOGEN_COMMENT = f"""// {_AUTOGEN_STR}
"""


_PYSTYLE_AUTOGEN_COMMENT = f"""# {_AUTOGEN_STR}
"""

_CUTLASS_KERNEL_ARGS_2x = """
  typename DeviceKernel::Arguments arguments {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},                                        // problem size
      1,
      {alpha, beta},
      A, B, C, D,
      0, 0, 0, 0,                                       // batch strides
      DeviceKernel::LayoutA::packed({M, K}).stride(0),  // lda
      DeviceKernel::LayoutB::packed({K, N}).stride(0),  // ldb
      DeviceKernel::LayoutC::packed({M, N}).stride(0),  // ldc
      DeviceKernel::LayoutC::packed({M, N}).stride(0)   // ldd
  };
"""

_CUTLASS_KERNEL_ARGS_2x_STREAM_K = """
  typename DeviceKernel::Arguments arguments {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K},                                        // problem size
      1,
      {alpha, beta},
      A, B, C, D,
      0, 0, 0, 0,                                       // batch strides
      DeviceKernel::LayoutA::packed({M, K}).stride(0),  // lda
      DeviceKernel::LayoutB::packed({K, N}).stride(0),  // ldb
      DeviceKernel::LayoutC::packed({M, N}).stride(0),  // ldc
      DeviceKernel::LayoutC::packed({M, N}).stride(0),  // ldd
      -1                                                // avail_sms
  };
"""

_CUTLASS_KERNEL_RUN_GEMM_2x = """
using ElementCompute = typename DeviceKernel::EpilogueOutputOp::ElementCompute;

cutlass::Status ${name}_kernel_run(int M, int N, int K,
                        const DeviceKernel::ElementA* A, const DeviceKernel::ElementB* B, const DeviceKernel::ElementC* C, DeviceKernel::ElementC* D,
                        ElementCompute alpha, ElementCompute beta) {
  ${args}
  size_t workspace_size = DeviceKernel::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  DeviceKernel gemm_op;
  cutlass::Status status = gemm_op.initialize(arguments,
                                              workspace.get(),
                                              nullptr);     // CUDA stream

  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm_op();
  return status;
}
"""

_CUTLASS_KERNEL_RUN_GEMM_3x = """
using StrideA = typename DeviceKernel::GemmKernel::StrideA;
using StrideB = typename DeviceKernel::GemmKernel::StrideB;
using StrideC = typename DeviceKernel::GemmKernel::StrideC;
using StrideD = typename DeviceKernel::GemmKernel::StrideD;

using ElementCompute = typename DeviceKernel::EpilogueOutputOp::ElementCompute;

cutlass::Status ${name}_kernel_run(
        int M, int N, int K, int L,
        const DeviceKernel::ElementA* A, const DeviceKernel::ElementB* B, const DeviceKernel::ElementC* C, DeviceKernel::ElementC* D,
        ElementCompute alpha, ElementCompute beta, const cutlass::KernelHardwareInfo& hw_info) {

  typename DeviceKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, L},                                                              // problem size
      {
        A,                                                                         // ptrA
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L)),    // stride A
        B,                                                                         // ptrB
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L)),    // stride B
      },
      {
        {alpha, beta},
        C,                                                                       // ptrC
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L)),  // stride C
        D,                                                                       // ptrD
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L)),  // stride D
      },
      hw_info
  };

  size_t workspace_size = DeviceKernel::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  DeviceKernel gemm_op;
  cutlass::Status status = gemm_op.run(arguments,
                                       workspace.get(),
                                       nullptr);     // CUDA stream

  return status;
}
"""


_CUTLASS_KERNEL_RUN_GROUPED_GEMM_2x = """
using ElementCompute = typename DeviceKernel::EpilogueOutputOp::ElementCompute;

int threadblock_count = DeviceKernel::sufficient();

cutlass::Status ${name}_kernel_run(int problem_count, cutlass::gemm::GemmCoord* problem_sizes,
                        DeviceKernel::ElementA** A, DeviceKernel::ElementB** B, DeviceKernel::ElementC** C, DeviceKernel::ElementC** D,
                        int64_t* lda, int64_t* ldb, int64_t* ldc, int64_t* ldd,
                        ElementCompute alpha, ElementCompute beta) {

  typename DeviceKernel::Arguments arguments {
    problem_sizes,
    problem_count,
    threadblock_count,
    {alpha, beta},
    A, B, C, D,
    lda, ldb, ldc, ldd
  };

  size_t workspace_size = DeviceKernel::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  DeviceKernel gemm_op;
  cutlass::Status status = gemm_op.initialize(arguments,
                                              workspace.get(),
                                              nullptr);     // CUDA stream

  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm_op();
  return status;
}
"""


_CUTLASS_KERNEL_RUN_CONV2D_2x = """

using UnderlyingKernel = typename DeviceKernel::UnderlyingKernel;
namespace {
using TensorRefA = typename UnderlyingKernel::TensorRefA;
using TensorRefB = typename UnderlyingKernel::TensorRefB;
using TensorRefC = typename UnderlyingKernel::TensorRefC;
using ElementCompute = typename UnderlyingKernel::EpilogueOutputOp::ElementCompute;
}

template<typename TensorRef, typename Element>
TensorRef get_tensor_ref(cutlass::Tensor4DCoord tensor_coord, Element* ptr){
  cutlass::layout::TensorNHWC layout = cutlass::layout::TensorNHWC::packed(tensor_coord);
  TensorRef tensor_ref(ptr, layout);
  return tensor_ref;
}

cutlass::Status ${name}_kernel_run(cutlass::conv::Conv2dProblemSize* problem_size,
                        UnderlyingKernel::ElementA* A, UnderlyingKernel::ElementB* B,
                        UnderlyingKernel::ElementC* C, UnderlyingKernel::ElementC* D,
                        ElementCompute alpha, ElementCompute beta, std::string split_k_mode,
                        cudaStream_t stream, int device_id=0) {
  // create the tensor references
  cutlass::Tensor4DCoord tensor_coord_A = cutlass::conv::implicit_gemm_tensor_a_extent(
    cutlass::conv::Operator::k${conv_kind_name}, *problem_size
  );
  cutlass::Tensor4DCoord tensor_coord_B = cutlass::conv::implicit_gemm_tensor_b_extent(
    cutlass::conv::Operator::k${conv_kind_name}, *problem_size
  );
  cutlass::Tensor4DCoord tensor_coord_C = cutlass::conv::implicit_gemm_tensor_c_extent(
    cutlass::conv::Operator::k${conv_kind_name}, *problem_size
  );

  TensorRefA tensor_ref_A = get_tensor_ref<TensorRefA, UnderlyingKernel::ElementA>(tensor_coord_A, A);
  TensorRefB tensor_ref_B = get_tensor_ref<TensorRefB, UnderlyingKernel::ElementB>(tensor_coord_B, B);
  TensorRefC tensor_ref_C = get_tensor_ref<TensorRefC, UnderlyingKernel::ElementC>(tensor_coord_C, C);
  TensorRefC tensor_ref_D = get_tensor_ref<TensorRefC, UnderlyingKernel::ElementC>(tensor_coord_C, D);

  cutlass::conv::SplitKMode mode;
  if (split_k_mode == "serial") {
    mode = cutlass::conv::SplitKMode::kSerial;
  } else if (split_k_mode == "parallel") {
    mode = cutlass::conv::SplitKMode::kParallel;
  } else {
    throw std::runtime_error("Invalid split_k_mode: " + split_k_mode);
  }

  typename DeviceKernel::Arguments arguments{
    *problem_size,
    tensor_ref_A,
    tensor_ref_B,
    tensor_ref_C,
    tensor_ref_D,
    {alpha, beta},
    mode
  };

  DeviceKernel implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  void* workspace_ptr = device_memory_allocation(workspace_size, device_id);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = implicit_gemm_op.initialize(arguments, workspace_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  //
  // Launch initialized CUTLASS kernel
  //
  status = implicit_gemm_op(stream);

  return status;
}
"""