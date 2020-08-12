// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#pragma once

#include "DLSystemView.h"

#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#endif


namespace dlext
{


namespace py = pybind11;

const char* const kDLTensorCapsuleName = "dltensor";


inline py::capsule encapsulate(DLManagedTensorPtr dl_managed_tensor)
{
    return py::capsule(
        dl_managed_tensor, kDLTensorCapsuleName,
        [](PyObject* obj) { // PyCapsule_Destructor
            DLManagedTensorPtr dlmt = static_cast<DLManagedTensorPtr>(
                PyCapsule_GetPointer(obj, kDLTensorCapsuleName)
            );
            if (dlmt && dlmt->deleter) {
                dlmt->deleter(dlmt);
            } else {
                PyErr_Clear();
            }
        }
    );
}


} // namespace dlext
