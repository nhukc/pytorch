#include <cuda_runtime.h>
#include <cuda.h>  // For cuFuncGetName
#include <iostream>
#include <vector>
#include <cassert>

// Simple CUDA kernel for testing
__global__ void test_kernel(float* data, int n, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= multiplier;
    }
}

// Another kernel with different signature for parameter update testing
__global__ void test_kernel2(float* data, int n, float addend) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += addend;
    }
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "Testing CUDA Graph Node Access APIs..." << std::endl;
    
    // Initialize data
    const int n = 1024;
    const size_t size = n * sizeof(float);
    
    float* h_data = new float[n];
    for (int i = 0; i < n; i++) {
        h_data[i] = i * 1.0f;
    }
    
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    // Create stream for graph capture
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Start graph capture
    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Launch kernel during capture
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    float multiplier = 2.0f;
    
    std::cout << "Launching kernel during capture..." << std::endl;
    std::cout << "Grid size: (" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << ")" << std::endl;
    std::cout << "Block size: (" << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << ")" << std::endl;
    
    test_kernel<<<gridSize, blockSize, 0, stream>>>(d_data, n, multiplier);
    
    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(kernelError) << std::endl;
    } else {
        std::cout << "Kernel launched successfully during capture" << std::endl;
    }
    
    // End capture
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    
    std::cout << "Graph captured successfully!" << std::endl;
    
    // Verify the graph is valid
    if (graph == nullptr) {
        std::cerr << "ERROR: Graph is null after capture!" << std::endl;
        return 1;
    } else {
        std::cout << "Graph pointer: " << graph << std::endl;
    }
    
    // Test 1: Get all nodes in the graph
    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &numNodes));
    std::cout << "Number of nodes in graph: " << numNodes << std::endl;
    
    std::vector<cudaGraphNode_t> nodes(numNodes);
    CHECK_CUDA(cudaGraphGetNodes(graph, nodes.data(), &numNodes));
    
    // Test 2: Get node types and find kernel nodes
    std::vector<cudaGraphNode_t> kernelNodes;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        CHECK_CUDA(cudaGraphNodeGetType(nodes[i], &nodeType));
        
        std::cout << "Node " << i << " type: ";
        switch (nodeType) {
            case cudaGraphNodeTypeKernel:
                std::cout << "Kernel";
                kernelNodes.push_back(nodes[i]);
                break;
            case cudaGraphNodeTypeMemcpy:
                std::cout << "Memcpy";
                break;
            case cudaGraphNodeTypeMemset:
                std::cout << "Memset";
                break;
            case cudaGraphNodeTypeHost:
                std::cout << "Host";
                break;
            case cudaGraphNodeTypeGraph:
                std::cout << "Graph";
                break;
            case cudaGraphNodeTypeEmpty:
                std::cout << "Empty";
                break;
            default:
                std::cout << "Unknown (" << nodeType << ")";
                break;
        }
        std::cout << std::endl;
    }
    
    std::cout << "Found " << kernelNodes.size() << " kernel nodes" << std::endl;
    
    // Test 3: Get kernel node parameters
    if (!kernelNodes.empty()) {
        cudaKernelNodeParams kernelParams;
        CHECK_CUDA(cudaGraphKernelNodeGetParams(kernelNodes[0], &kernelParams));
        
        std::cout << "Kernel node parameters:" << std::endl;
        std::cout << "  func: " << kernelParams.func << std::endl;
        std::cout << "  gridDim: (" << kernelParams.gridDim.x << ", " << kernelParams.gridDim.y << ", " << kernelParams.gridDim.z << ")" << std::endl;
        std::cout << "  blockDim: (" << kernelParams.blockDim.x << ", " << kernelParams.blockDim.y << ", " << kernelParams.blockDim.z << ")" << std::endl;
        std::cout << "  sharedMemBytes: " << kernelParams.sharedMemBytes << std::endl;
        std::cout << "  kernelParams: " << kernelParams.kernelParams << std::endl;
        std::cout << "  extra: " << kernelParams.extra << std::endl;
        
        // Test automatic kernel naming using driver API
        std::cout << "\nTesting automatic kernel naming:" << std::endl;
        
        // Initialize CUDA driver API
        CUresult init_result = cuInit(0);
        if (init_result == CUDA_SUCCESS) {
            // Convert node handle (cudaGraphNode_t and CUgraphNode are the same type)
            CUgraphNode cu_node = reinterpret_cast<CUgraphNode>(kernelNodes[0]);
            
            // Get parameters using driver API
            CUDA_KERNEL_NODE_PARAMS cu_kernel_params;
            if (cuGraphKernelNodeGetParams(cu_node, &cu_kernel_params) == CUDA_SUCCESS) {
                // Extract function name
                const char* func_name = nullptr;
                if (cuFuncGetName(&func_name, cu_kernel_params.func) == CUDA_SUCCESS && func_name) {
                    std::cout << "  SUCCESS: Automatic kernel name: '" << func_name << "'" << std::endl;
                } else {
                    std::cout << "  Could not get function name - would use fallback" << std::endl;
                }
            } else {
                std::cout << "  Could not get kernel parameters" << std::endl;
            }
        } else {
            std::cout << "  Could not initialize CUDA driver API" << std::endl;
        }
    }
    
    // Test 4: Create executable graph and test parameter updates
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    
    std::cout << "Graph instantiated successfully!" << std::endl;
    
    // Test 5: Update kernel node parameters
    if (!kernelNodes.empty()) {
        // Create new parameters for the kernel
        float new_multiplier = 3.0f;
        int n_copy = n;  // Make a non-const copy
        void* new_args[] = {&d_data, &n_copy, &new_multiplier};
        
        cudaKernelNodeParams newParams;
        newParams.func = (void*)test_kernel;
        newParams.gridDim = gridSize;
        newParams.blockDim = blockSize;
        newParams.sharedMemBytes = 0;
        newParams.kernelParams = new_args;
        newParams.extra = nullptr;
        
        CHECK_CUDA(cudaGraphExecKernelNodeSetParams(graphExec, kernelNodes[0], &newParams));
        std::cout << "Updated kernel node parameters successfully!" << std::endl;
    }
    
    // Test 6: Launch the graph and verify results
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    
    // Copy result back and verify
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    
    // Check if the multiplication worked (should be 3.0 * original value due to parameter update)
    bool correct = true;
    for (int i = 0; i < 10; i++) {  // Check first 10 elements
        float expected = i * 3.0f;  // 3.0 due to parameter update
        if (abs(h_data[i] - expected) > 1e-6) {
            std::cout << "Mismatch at index " << i << ": expected " << expected << ", got " << h_data[i] << std::endl;
            correct = false;
        }
    }
    
    if (correct) {
        std::cout << "Graph execution with updated parameters: SUCCESS!" << std::endl;
    } else {
        std::cout << "Graph execution with updated parameters: FAILED!" << std::endl;
    }
    
    // Test 7: Test swapping pointer arguments
    std::cout << "\n=== Testing Pointer Argument Swapping ===" << std::endl;
    
    // Create a second data buffer
    float* d_data2;
    CHECK_CUDA(cudaMalloc(&d_data2, size));
    
    // Initialize second buffer with different values (starting from 100)
    float* h_data2 = new float[n];
    for (int i = 0; i < n; i++) {
        h_data2[i] = 100.0f + i;
    }
    CHECK_CUDA(cudaMemcpy(d_data2, h_data2, size, cudaMemcpyHostToDevice));
    
    if (!kernelNodes.empty()) {
        // Update kernel to use the second buffer instead of the first
        float ptr_test_multiplier = 2.0f;
        int n_ptr_copy = n;
        void* ptr_swap_args[] = {&d_data2, &n_ptr_copy, &ptr_test_multiplier};  // Note: d_data2 instead of d_data
        
        cudaKernelNodeParams ptrSwapParams;
        ptrSwapParams.func = (void*)test_kernel;
        ptrSwapParams.gridDim = gridSize;
        ptrSwapParams.blockDim = blockSize;
        ptrSwapParams.sharedMemBytes = 0;
        ptrSwapParams.kernelParams = ptr_swap_args;
        ptrSwapParams.extra = nullptr;
        
        CHECK_CUDA(cudaGraphExecKernelNodeSetParams(graphExec, kernelNodes[0], &ptrSwapParams));
        std::cout << "Updated kernel to use second buffer (d_data2)" << std::endl;
        
        // Launch graph with swapped pointer
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Copy second buffer back and verify
        CHECK_CUDA(cudaMemcpy(h_data2, d_data2, size, cudaMemcpyDeviceToHost));
        
        // Check if the multiplication worked on the second buffer
        bool ptr_swap_correct = true;
        std::cout << "Checking results from second buffer:" << std::endl;
        for (int i = 0; i < 5; i++) {  // Check first 5 elements
            float expected = (100.0f + i) * 2.0f;  // Original value * multiplier
            std::cout << "  Index " << i << ": expected " << expected << ", got " << h_data2[i] << std::endl;
            if (abs(h_data2[i] - expected) > 1e-6) {
                ptr_swap_correct = false;
            }
        }
        
        if (ptr_swap_correct) {
            std::cout << "Pointer argument swapping: SUCCESS!" << std::endl;
        } else {
            std::cout << "Pointer argument swapping: FAILED!" << std::endl;
        }
        
        // Verify first buffer was not modified
        CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
        bool first_buffer_unchanged = true;
        std::cout << "Checking first buffer remained unchanged:" << std::endl;
        for (int i = 0; i < 5; i++) {
            float expected = i * 3.0f;  // Should still have results from previous test
            std::cout << "  Index " << i << ": expected " << expected << ", got " << h_data[i] << std::endl;
            if (abs(h_data[i] - expected) > 1e-6) {
                first_buffer_unchanged = false;
            }
        }
        
        if (first_buffer_unchanged) {
            std::cout << "First buffer unchanged: SUCCESS!" << std::endl;
        } else {
            std::cout << "First buffer unchanged: FAILED!" << std::endl;
        }
    }
    
    // Cleanup additional resources
    CHECK_CUDA(cudaFree(d_data2));
    delete[] h_data2;
    
    // Cleanup original resources
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_data));
    delete[] h_data;
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}