// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef PY_HOOMD_DLPACK_EXTENSION_H_
#define PY_HOOMD_DLPACK_EXTENSION_H_

#include "DLExt.h"
#ifdef HOOMD2
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#else
#include <pybind11/pybind11.h>
#endif

namespace hoomd
{
namespace md
{
namespace dlext
{

using PyCapsule = pybind11::capsule;
using PyTensorBundle = std::tuple<PyObject*, DLManagedTensor*, DLManagedTensorDeleter>;

const char* const kDLTensorCapsuleName = "dltensor";
const char* const kUsedDLTensorCapsuleName = "used_dltensor";

static std::vector<PyTensorBundle> kPyCapsulesPool;

inline PyCapsule enpycapsulate(DLManagedTensor* tensor, bool autodestruct = true)
{
    auto capsule = PyCapsule(tensor, kDLTensorCapsuleName);  // default destructor is nullptr
    if (autodestruct)
        PyCapsule_SetDestructor(
            capsule.ptr(),
            [](PyObject* obj) {  // PyCapsule_Destructor
                auto dlmt = static_cast<DLManagedTensor*>(
                    PyCapsule_GetPointer(obj, kDLTensorCapsuleName)
                );
                if (dlmt && dlmt->deleter) {
                    dlmt->deleter(dlmt);
                } else {
                    PyErr_Clear();
                }
            }
        );
    return capsule;
}

template <typename Property>
struct DEFAULT_VISIBILITY PyUnsafeEncapsulator final {
    static PyCapsule wrap(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        DLManagedTensor* tensor = Property::from(sysview, location, mode);
        return enpycapsulate(tensor);
    }
};

template <typename Property>
struct DEFAULT_VISIBILITY PyEncapsulator final {
    static PyCapsule wrap(
        SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        if (!sysview.in_context_manager())
            throw std::runtime_error("Cannot access property outside a context manager.");
        auto tensor = Property::from(sysview, location, mode);
        auto capsule = enpycapsulate(tensor, /* autodestruct = */ false);
        kPyCapsulesPool.push_back(std::make_tuple(capsule.ptr(), tensor, tensor->deleter));
        // We manually delete the tensor when exiting the context manager,
        // so we need to prevent others from grabbing the default deleter.
        tensor->deleter = do_not_delete;
        return capsule;
    }
};

void invalidate(PyTensorBundle& bundle)
{
    auto obj = std::get<0>(bundle);
    auto tensor = std::get<1>(bundle);
    auto shred = std::get<2>(bundle);

    shred(tensor);

    if (PyCapsule_IsValid(obj, kDLTensorCapsuleName)) {
        PyCapsule_SetName(obj, kUsedDLTensorCapsuleName);
        PyCapsule_SetPointer(obj, opaque(&kInvalidDLManagedTensor));
    } else if (PyCapsule_IsValid(obj, kUsedDLTensorCapsuleName)) {
        PyCapsule_SetPointer(obj, opaque(&kInvalidDLManagedTensor));
    }
}

}  // namespace dlext
}  // namespace md
}  // namespace hoomd

#endif  // PY_HOOMD_DLPACK_EXTENSION_H_
