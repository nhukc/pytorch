#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/flat_hash_map.h>

#include <cuda.h>  // For driver API types and functions
#include <vector>
#include <string>
#include <memory>

namespace at {
namespace cuda {

// Wrapper for individual CUDA graph nodes
struct TORCH_CUDA_CPP_API CUDAGraphNode {
  CUDAGraphNode(cudaGraphNode_t node, cudaGraphNodeType type);
  ~CUDAGraphNode() = default;

  // Node information
  cudaGraphNode_t node() const { return node_; }
  cudaGraphNodeType type() const { return type_; }
  const std::string& name() const { return name_; }
  void set_name(const std::string& name) { name_ = name; }
  
  // Type checking
  bool isKernelNode() const { return type_ == cudaGraphNodeTypeKernel; }
  
  // Get automatic kernel name (requires driver API)
  std::string getAutomaticKernelName() const;

  // Kernel parameter access (safe, read-only)
  dim3 getKernelGridDim() const;
  dim3 getKernelBlockDim() const;
  unsigned int getKernelSharedMemBytes() const;
  size_t getKernelNumArgs() const;
  void* getKernelArg(size_t index) const;

private:
  cudaGraphNode_t node_;
  cudaGraphNodeType type_;
  std::string name_;
  
  // Helper for automatic naming
  static bool driver_api_initialized_;
  static void ensureDriverApiInitialized();
  
  // Internal helper to get full kernel params
  cudaKernelNodeParams getKernelParamsInternal() const;
  
  // Friend class to allow access to private methods
  friend class CUDAGraphWithNodes;
};

// Extended CUDA graph with node access and manipulation
struct TORCH_CUDA_CPP_API CUDAGraphWithNodes : public CUDAGraph {
  CUDAGraphWithNodes();
  ~CUDAGraphWithNodes();

  // Override capture_end to collect node information
  void capture_end();
  
  // Node access methods
  size_t num_nodes() const;
  std::vector<std::shared_ptr<CUDAGraphNode>> get_nodes() const;
  std::shared_ptr<CUDAGraphNode> get_node(size_t index) const;
  std::shared_ptr<CUDAGraphNode> get_node_by_name(const std::string& name) const;
  
  // Node filtering methods
  std::vector<std::shared_ptr<CUDAGraphNode>> get_kernel_nodes() const;
  std::vector<std::shared_ptr<CUDAGraphNode>> get_nodes_by_type(cudaGraphNodeType type) const;
  
  // Safe kernel parameter updates (no raw void** exposure)
  void update_kernel_grid(size_t node_index, dim3 grid_dim, dim3 block_dim);
  void update_kernel_grid(const std::string& node_name, dim3 grid_dim, dim3 block_dim);
  
  void update_kernel_shared_mem(size_t node_index, unsigned int shared_mem_bytes);
  void update_kernel_shared_mem(const std::string& node_name, unsigned int shared_mem_bytes);
  
  // Simple tensor pointer updates (preserve all other arguments)
  void update_kernel_tensor_pointer(size_t node_index, size_t arg_index, const at::Tensor& tensor);
  void update_kernel_tensor_pointer(const std::string& node_name, size_t arg_index, const at::Tensor& tensor);
  
  // Node naming (optional, for easier identification)
  void set_node_name(size_t node_index, const std::string& name);
  
  // Get node names (automatic + manual)
  std::vector<std::string> get_node_names() const;

private:
  void collect_graph_nodes();
  void clear_nodes();
  
  // Internal helper for parameter updates
  void update_kernel_node_params_internal(cudaGraphNode_t node, const cudaKernelNodeParams& params);
  
  std::vector<std::shared_ptr<CUDAGraphNode>> nodes_;
  ska::flat_hash_map<std::string, std::shared_ptr<CUDAGraphNode>> named_nodes_;
  
  // Track allocated memory for cleanup
  std::vector<void**> allocated_arg_arrays_;
};

} // namespace cuda
} // namespace at