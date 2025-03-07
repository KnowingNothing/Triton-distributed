//===- op_pybind.h ----------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>
#include <torch/python.h>

namespace py = pybind11;

namespace distributed {

// Registry of functions that register
// functions into module
class OpInitRegistry {
public:
  using OpInitFunc = std::function<void(py::module &)>;
  static OpInitRegistry &instance();
  void register_one(std::string name, OpInitFunc &&func);
  void initialize_all(py::module &m) const;

private:
  std::map<std::string, OpInitFunc> registry_;
  mutable std::mutex register_mutex_;

  OpInitRegistry() {}
  OpInitRegistry(const OpInitRegistry &) = delete;
  OpInitRegistry &operator=(const OpInitRegistry &) = delete;
};

template <typename T>
struct TorchClassWrapper : public torch::CustomClassHolder, T {
public:
  using T::T;
};

} // namespace distributed