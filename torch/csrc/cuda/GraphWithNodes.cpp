#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>
#include <pybind11/stl.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraphWithNodes.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraphWithNodes_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();

  // Bind cudaGraphNodeType enum
  py::enum_<cudaGraphNodeType>(torch_C_m, "cudaGraphNodeType")
      .value("cudaGraphNodeTypeKernel", cudaGraphNodeTypeKernel)
      .value("cudaGraphNodeTypeMemcpy", cudaGraphNodeTypeMemcpy)
      .value("cudaGraphNodeTypeMemset", cudaGraphNodeTypeMemset)
      .value("cudaGraphNodeTypeHost", cudaGraphNodeTypeHost)
      .value("cudaGraphNodeTypeGraph", cudaGraphNodeTypeGraph)
      .value("cudaGraphNodeTypeEmpty", cudaGraphNodeTypeEmpty)
      .export_values();

  // Bind dim3 struct
  py::class_<dim3>(torch_C_m, "dim3")
      .def(py::init<unsigned int, unsigned int, unsigned int>())
      .def_readwrite("x", &dim3::x)
      .def_readwrite("y", &dim3::y)
      .def_readwrite("z", &dim3::z)
      .def("__repr__", [](const dim3& d) {
          return "dim3(" + std::to_string(d.x) + ", " + std::to_string(d.y) + ", " + std::to_string(d.z) + ")";
      });

  // Bind CUDAGraphNode
  shared_ptr_class_<::at::cuda::CUDAGraphNode>(torch_C_m, "_CUDAGraphNode")
      // Note: Skip binding node() method since CUgraphNode_st is incomplete type
      .def("type", &::at::cuda::CUDAGraphNode::type)
      .def("name", &::at::cuda::CUDAGraphNode::name)
      .def("set_name", &::at::cuda::CUDAGraphNode::set_name, py::arg("name"))
      .def("isKernelNode", &::at::cuda::CUDAGraphNode::isKernelNode)
      .def(
          "getAutomaticKernelName",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphNode::getAutomaticKernelName))
      .def(
          "getKernelGridDim",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphNode::getKernelGridDim))
      .def(
          "getKernelBlockDim", 
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphNode::getKernelBlockDim))
      .def(
          "getKernelSharedMemBytes",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphNode::getKernelSharedMemBytes))
      .def(
          "getKernelNumArgs",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphNode::getKernelNumArgs))
      .def(
          "getKernelArg",
          [](const ::at::cuda::CUDAGraphNode& self, size_t index) -> uintptr_t {
            void* ptr = self.getKernelArg(index);
            return reinterpret_cast<uintptr_t>(ptr);
          },
          py::arg("index"));

  // Bind CUDAGraphWithNodes (inherits from CUDAGraph)
  py::class_<::at::cuda::CUDAGraphWithNodes, ::at::cuda::CUDAGraph, std::shared_ptr<::at::cuda::CUDAGraphWithNodes>>(
      torch_C_m, "_CUDAGraphWithNodes")
      .def(py::init<>())
      // Override capture_begin to accept derived class
      .def(
          "capture_begin",
          [](::at::cuda::CUDAGraphWithNodes& self,
             std::optional<c10::cuda::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            cudaStreamCaptureMode capture_mode{};
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::cuda::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool") = py::none(),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      // Override capture_end to use derived class method
      .def(
          "capture_end",
          [](::at::cuda::CUDAGraphWithNodes& self) {
            py::gil_scoped_release release;
            return self.capture_end();
          })
      // Node access methods
      .def(
          "num_nodes",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::num_nodes))
      .def(
          "get_nodes",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::get_nodes))
      .def(
          "get_node",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::get_node),
          py::arg("index"))
      .def(
          "get_node_by_name",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::get_node_by_name),
          py::arg("name"))
      .def(
          "get_kernel_nodes",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::get_kernel_nodes))
      .def(
          "get_nodes_by_type",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::get_nodes_by_type),
          py::arg("node_type"))
      .def(
          "get_node_names",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::get_node_names))
      
      // Node naming
      .def(
          "set_node_name",
          torch::wrap_pybind_function_no_gil(&::at::cuda::CUDAGraphWithNodes::set_node_name),
          py::arg("node_index"),
          py::arg("name"))
      
      // Kernel parameter updates - grid/block dimensions
      .def(
          "update_kernel_grid",
          [](::at::cuda::CUDAGraphWithNodes& self, size_t node_index, py::tuple grid_dim, py::tuple block_dim) {
            // Convert Python tuples to dim3
            TORCH_CHECK(grid_dim.size() <= 3, "Grid dimension tuple must have at most 3 elements");
            TORCH_CHECK(block_dim.size() <= 3, "Block dimension tuple must have at most 3 elements");
            
            dim3 grid(1, 1, 1);
            dim3 block(1, 1, 1);
            
            if (grid_dim.size() >= 1) grid.x = grid_dim[0].cast<unsigned int>();
            if (grid_dim.size() >= 2) grid.y = grid_dim[1].cast<unsigned int>();
            if (grid_dim.size() >= 3) grid.z = grid_dim[2].cast<unsigned int>();
            
            if (block_dim.size() >= 1) block.x = block_dim[0].cast<unsigned int>();
            if (block_dim.size() >= 2) block.y = block_dim[1].cast<unsigned int>();
            if (block_dim.size() >= 3) block.z = block_dim[2].cast<unsigned int>();
            
            py::gil_scoped_release release;
            return self.update_kernel_grid(node_index, grid, block);
          },
          py::arg("node_index"),
          py::arg("grid_dim"),
          py::arg("block_dim"))
      .def(
          "update_kernel_grid",
          [](::at::cuda::CUDAGraphWithNodes& self, const std::string& node_name, py::tuple grid_dim, py::tuple block_dim) {
            // Convert Python tuples to dim3
            TORCH_CHECK(grid_dim.size() <= 3, "Grid dimension tuple must have at most 3 elements");
            TORCH_CHECK(block_dim.size() <= 3, "Block dimension tuple must have at most 3 elements");
            
            dim3 grid(1, 1, 1);
            dim3 block(1, 1, 1);
            
            if (grid_dim.size() >= 1) grid.x = grid_dim[0].cast<unsigned int>();
            if (grid_dim.size() >= 2) grid.y = grid_dim[1].cast<unsigned int>();
            if (grid_dim.size() >= 3) grid.z = grid_dim[2].cast<unsigned int>();
            
            if (block_dim.size() >= 1) block.x = block_dim[0].cast<unsigned int>();
            if (block_dim.size() >= 2) block.y = block_dim[1].cast<unsigned int>();
            if (block_dim.size() >= 3) block.z = block_dim[2].cast<unsigned int>();
            
            py::gil_scoped_release release;
            return self.update_kernel_grid(node_name, grid, block);
          },
          py::arg("node_name"),
          py::arg("grid_dim"),
          py::arg("block_dim"))
      
      // Shared memory updates
      .def(
          "update_kernel_shared_mem",
          static_cast<void (::at::cuda::CUDAGraphWithNodes::*)(size_t, unsigned int)>(&::at::cuda::CUDAGraphWithNodes::update_kernel_shared_mem),
          py::arg("node_index"),
          py::arg("shared_mem_bytes"))
      .def(
          "update_kernel_shared_mem",
          static_cast<void (::at::cuda::CUDAGraphWithNodes::*)(const std::string&, unsigned int)>(&::at::cuda::CUDAGraphWithNodes::update_kernel_shared_mem),
          py::arg("node_name"),
          py::arg("shared_mem_bytes"))
      
      // Simple tensor pointer updates (preserve all other arguments)
      .def(
          "update_kernel_tensor_pointer",
          static_cast<void (::at::cuda::CUDAGraphWithNodes::*)(size_t, size_t, const at::Tensor&)>(&::at::cuda::CUDAGraphWithNodes::update_kernel_tensor_pointer),
          py::arg("node_index"),
          py::arg("arg_index"),
          py::arg("tensor"))
      .def(
          "update_kernel_tensor_pointer",
          static_cast<void (::at::cuda::CUDAGraphWithNodes::*)(const std::string&, size_t, const at::Tensor&)>(&::at::cuda::CUDAGraphWithNodes::update_kernel_tensor_pointer),
          py::arg("node_name"),
          py::arg("arg_index"),
          py::arg("tensor"));
}