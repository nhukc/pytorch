#include <ATen/cuda/CUDAGraphWithNodes.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cstddef>
#include <iostream>

namespace at::cuda {

// Static member initialization
bool CUDAGraphNode::driver_api_initialized_ = false;

// CUDAGraphNode implementation
CUDAGraphNode::CUDAGraphNode(cudaGraphNode_t node, cudaGraphNodeType type) 
    : node_(node), type_(type) {
}

void CUDAGraphNode::ensureDriverApiInitialized() {
    if (!driver_api_initialized_) {
        CUresult result = cuInit(0);
        if (result == CUDA_SUCCESS) {
            driver_api_initialized_ = true;
        }
    }
}

std::string CUDAGraphNode::getAutomaticKernelName() const {
    if (!isKernelNode()) {
        return "";
    }
    
    ensureDriverApiInitialized();
    if (!driver_api_initialized_) {
        return "";
    }
    
    try {
        // Convert to driver API node handle
        CUgraphNode cu_node = reinterpret_cast<CUgraphNode>(node_);
        
        // Get parameters using driver API
        CUDA_KERNEL_NODE_PARAMS cu_kernel_params;
        CUresult params_result = cuGraphKernelNodeGetParams(cu_node, &cu_kernel_params);
        
        if (params_result == CUDA_SUCCESS) {
            // Extract function name
            const char* func_name = nullptr;
            CUresult name_result = cuFuncGetName(&func_name, cu_kernel_params.func);
            
            if (name_result == CUDA_SUCCESS && func_name != nullptr) {
                return std::string(func_name);
            }
        }
    } catch (...) {
        // Fallback on any error
    }
    
    return "";
}

cudaKernelNodeParams CUDAGraphNode::getKernelParamsInternal() const {
    TORCH_CHECK(isKernelNode(), "Node is not a kernel node");
    
    cudaKernelNodeParams params;
    AT_CUDA_CHECK(cudaGraphKernelNodeGetParams(node_, &params));
    
    return params;
}

dim3 CUDAGraphNode::getKernelGridDim() const {
    auto params = getKernelParamsInternal();
    return params.gridDim;
}

dim3 CUDAGraphNode::getKernelBlockDim() const {
    auto params = getKernelParamsInternal();
    return params.blockDim;
}

unsigned int CUDAGraphNode::getKernelSharedMemBytes() const {
    auto params = getKernelParamsInternal();
    return params.sharedMemBytes;
}

size_t CUDAGraphNode::getKernelNumArgs() const {
    TORCH_CHECK(isKernelNode(), "Node is not a kernel node");
    
    auto params = getKernelParamsInternal();
    if (params.kernelParams == nullptr) {
        return 0;
    }
    
    // Examine the argument array to determine how many arguments there are
    void** args = (void**)params.kernelParams;
    size_t count = 0;
    
    // Use heuristics to count arguments - look for reasonable patterns
    // Most kernels have fewer than 16 arguments
    for (size_t i = 0; i < 16; i++) {
        void* arg = args[i];
        
        // Stop counting if we hit a null pointer (common terminator)
        if (arg == nullptr) {
            break;
        }
        
        // Continue counting - this is a valid argument
        count++;
        
        // Also stop if we see a pattern that suggests end of arguments
        // (This is heuristic-based and may need refinement)
        if (i > 0 && args[i] == nullptr && args[i-1] != nullptr) {
            break;
        }
    }
    
    return count;
}

void* CUDAGraphNode::getKernelArg(size_t index) const {
    TORCH_CHECK(isKernelNode(), "Node is not a kernel node");
    
    auto params = getKernelParamsInternal();
    TORCH_CHECK(params.kernelParams != nullptr, "Kernel parameters array is null");
    
    // Basic bounds check
    TORCH_CHECK(index < 16, "Argument index too large (max 16 supported)");
    
    void** args = (void**)params.kernelParams;
    return args[index];
}

// CUDAGraphWithNodes implementation
CUDAGraphWithNodes::CUDAGraphWithNodes() : CUDAGraph() {
}

CUDAGraphWithNodes::~CUDAGraphWithNodes() {
    // Clean up allocated argument arrays
    for (void** array : allocated_arg_arrays_) {
        delete[] array;
    }
    allocated_arg_arrays_.clear();
    
    // Clean up the graph that we kept alive for parameter access
    // The base class destructor will handle graph_exec_ cleanup
    if (has_graph_ && graph_ != nullptr) {
        AT_CUDA_CHECK(cudaGraphDestroy(graph_));
        has_graph_ = false;
    }
}

void CUDAGraphWithNodes::capture_end() {
    // Get current stream and perform capture end operations up to graph creation
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(stream == capture_stream_,
                "Capture must end on the same stream it began on.");

    AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
    c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);
    TORCH_CHECK(graph_ != nullptr, "Invalid capture.");
    has_graph_ = true;

    // Now continue with the rest of the parent implementation
    // (graph instantiation, generator state handling, etc.)
    
    // Graph instantiation code from parent
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
    int version = 0;
    AT_CUDA_CHECK(cudaDriverGetVersion(&version));
    if (version < 11040) {
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
        AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
        AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
    } else {
        AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                    graph_,
                                                    cudaGraphInstantiateFlagAutoFreeOnLaunch));
    }
#endif

    has_graph_exec_ = true;

    // Collect node information AFTER graph instantiation (when parameters are fully available)
    collect_graph_nodes();

    // Generator state handling from parent
    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
        wholegraph_increments = generator_state->capture_epilogue();
    }

    // Check for empty graph warning
    size_t numCUDAGraphNodes = 0;
    AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
    if (numCUDAGraphNodes == 0) {
        TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                   "attempted to be captured on wrong device or stream.");
    }

    // Graph cleanup from parent (check debug flag) 
    // Note: _cuda_graphs_debug is defined in CUDAGraph.cpp
    // IMPORTANT: We DON'T destroy the graph because we need to keep node references valid
    // for parameter access. This is different from the base class behavior.
    // TODO: Consider extracting all needed data in collect_graph_nodes() instead
    // For now, keep the graph alive so node references remain valid
    // AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    // has_graph_ = false;
    // Keep graph alive so node references remain valid for parameter access
}

void CUDAGraphWithNodes::collect_graph_nodes() {
    clear_nodes();
    
    // Use graph_ instead of has_graph_exec_ since graph_ exists after capture ends
    // but before instantiation completes in the parent capture_end()
    if (graph_ == nullptr) {
        return;
    }
    
    // Get all nodes in the graph
    size_t numNodes = 0;
    AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numNodes));
    
    if (numNodes == 0) {
        return;
    }
    
    std::vector<cudaGraphNode_t> cuda_nodes(numNodes);
    AT_CUDA_CHECK(cudaGraphGetNodes(graph_, cuda_nodes.data(), &numNodes));
    
    // Create wrapper nodes
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        AT_CUDA_CHECK(cudaGraphNodeGetType(cuda_nodes[i], &nodeType));
        
        auto node_wrapper = std::make_shared<CUDAGraphNode>(cuda_nodes[i], nodeType);
        
        // Try automatic naming for kernel nodes
        if (nodeType == cudaGraphNodeTypeKernel) {
            std::string auto_name = node_wrapper->getAutomaticKernelName();
            if (auto_name.empty()) {
                node_wrapper->set_name("kernel_" + std::to_string(i));
            } else {
                node_wrapper->set_name(auto_name);
            }
        } else {
            // Name other node types by their type
            std::string type_name;
            switch (nodeType) {
                case cudaGraphNodeTypeMemcpy:
                    type_name = "memcpy_" + std::to_string(i);
                    break;
                case cudaGraphNodeTypeMemset:
                    type_name = "memset_" + std::to_string(i);
                    break;
                case cudaGraphNodeTypeHost:
                    type_name = "host_" + std::to_string(i);
                    break;
                case cudaGraphNodeTypeGraph:
                    type_name = "graph_" + std::to_string(i);
                    break;
                case cudaGraphNodeTypeEmpty:
                    type_name = "empty_" + std::to_string(i);
                    break;
                default:
                    type_name = "node_" + std::to_string(i);
                    break;
            }
            node_wrapper->set_name(type_name);
        }
        
        nodes_.push_back(node_wrapper);
        named_nodes_[node_wrapper->name()] = node_wrapper;
    }
}

void CUDAGraphWithNodes::clear_nodes() {
    nodes_.clear();
    named_nodes_.clear();
}

size_t CUDAGraphWithNodes::num_nodes() const {
    return nodes_.size();
}

std::vector<std::shared_ptr<CUDAGraphNode>> CUDAGraphWithNodes::get_nodes() const {
    return nodes_;
}

std::shared_ptr<CUDAGraphNode> CUDAGraphWithNodes::get_node(size_t index) const {
    TORCH_CHECK(index < nodes_.size(), "Node index out of range");
    return nodes_[index];
}

std::shared_ptr<CUDAGraphNode> CUDAGraphWithNodes::get_node_by_name(const std::string& name) const {
    auto it = named_nodes_.find(name);
    TORCH_CHECK(it != named_nodes_.end(), "Node with name '", name, "' not found");
    return it->second;
}

std::vector<std::shared_ptr<CUDAGraphNode>> CUDAGraphWithNodes::get_kernel_nodes() const {
    return get_nodes_by_type(cudaGraphNodeTypeKernel);
}

std::vector<std::shared_ptr<CUDAGraphNode>> CUDAGraphWithNodes::get_nodes_by_type(cudaGraphNodeType type) const {
    std::vector<std::shared_ptr<CUDAGraphNode>> filtered;
    for (const auto& node : nodes_) {
        if (node->type() == type) {
            filtered.push_back(node);
        }
    }
    return filtered;
}

void CUDAGraphWithNodes::set_node_name(size_t node_index, const std::string& name) {
    auto node = get_node(node_index);
    
    // Remove old name from map
    named_nodes_.erase(node->name());
    
    // Set new name
    node->set_name(name);
    named_nodes_[name] = node;
}

std::vector<std::string> CUDAGraphWithNodes::get_node_names() const {
    std::vector<std::string> names;
    for (const auto& node : nodes_) {
        names.push_back(node->name());
    }
    return names;
}

void CUDAGraphWithNodes::update_kernel_node_params_internal(cudaGraphNode_t node, const cudaKernelNodeParams& params) {
    TORCH_CHECK(has_graph_exec_, "Graph must be captured before updating parameters");
    AT_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph_exec_, node, &params));
}

void CUDAGraphWithNodes::update_kernel_grid(size_t node_index, dim3 grid_dim, dim3 block_dim) {
    auto node = get_node(node_index);
    TORCH_CHECK(node->isKernelNode(), "Node is not a kernel node");
    
    // Get current parameters
    auto params = node->getKernelParamsInternal();
    
    // Update grid dimensions
    params.gridDim = grid_dim;
    params.blockDim = block_dim;
    
    // Apply changes
    update_kernel_node_params_internal(node->node(), params);
}

void CUDAGraphWithNodes::update_kernel_grid(const std::string& node_name, dim3 grid_dim, dim3 block_dim) {
    auto node = get_node_by_name(node_name);
    TORCH_CHECK(node->isKernelNode(), "Node is not a kernel node");
    
    // Get current parameters
    auto params = node->getKernelParamsInternal();
    
    // Update grid dimensions
    params.gridDim = grid_dim;
    params.blockDim = block_dim;
    
    // Apply changes
    update_kernel_node_params_internal(node->node(), params);
}

void CUDAGraphWithNodes::update_kernel_shared_mem(size_t node_index, unsigned int shared_mem_bytes) {
    auto node = get_node(node_index);
    TORCH_CHECK(node->isKernelNode(), "Node is not a kernel node");
    
    // Get current parameters
    auto params = node->getKernelParamsInternal();
    
    // Update shared memory
    params.sharedMemBytes = shared_mem_bytes;
    
    // Apply changes
    update_kernel_node_params_internal(node->node(), params);
}

void CUDAGraphWithNodes::update_kernel_shared_mem(const std::string& node_name, unsigned int shared_mem_bytes) {
    auto node = get_node_by_name(node_name);
    TORCH_CHECK(node->isKernelNode(), "Node is not a kernel node");
    
    // Get current parameters
    auto params = node->getKernelParamsInternal();
    
    // Update shared memory
    params.sharedMemBytes = shared_mem_bytes;
    
    // Apply changes
    update_kernel_node_params_internal(node->node(), params);
}

void CUDAGraphWithNodes::update_kernel_tensor_pointer(size_t node_index, size_t arg_index, const at::Tensor& tensor) {
    auto node = get_node(node_index);
    TORCH_CHECK(node->isKernelNode(), "Node is not a kernel node");
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    
    // Get the current complete parameter structure
    auto params = node->getKernelParamsInternal();
    TORCH_CHECK(params.kernelParams != nullptr, "Kernel parameters array is null");
    
    // Get the argument array and replace just the specified pointer
    void** args = (void**)params.kernelParams;
    void* new_ptr = tensor.data_ptr();
    
    // Simple direct update - just change the one pointer we care about
    args[arg_index] = new_ptr;
    
    // Use the exact same parameters structure, no copying
    AT_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph_exec_, node->node(), &params));
}

void CUDAGraphWithNodes::update_kernel_tensor_pointer(const std::string& node_name, size_t arg_index, const at::Tensor& tensor) {
    auto node = get_node_by_name(node_name);
    TORCH_CHECK(node->isKernelNode(), "Node is not a kernel node");
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    
    // Get the current complete parameter structure
    auto params = node->getKernelParamsInternal();
    TORCH_CHECK(params.kernelParams != nullptr, "Kernel parameters array is null");
    
    // Get the argument array and replace just the specified pointer
    void** args = (void**)params.kernelParams;
    void* new_ptr = tensor.data_ptr();
    
    // Simple direct update - just change the one pointer we care about
    args[arg_index] = new_ptr;
    
    // Use the exact same parameters structure, no copying
    AT_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph_exec_, node->node(), &params));
}

} // namespace at::cuda