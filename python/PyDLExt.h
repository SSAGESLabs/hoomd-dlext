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

const char* const kDLTensorCapsuleName = "dltensor";
const char* const kUsedDLTensorCapsuleName = "used_dltensor";

class DEFAULT_VISIBILITY PyDLPackTensor final {
public:
    // Owning: the capsule's destructor frees the tensor.
    explicit PyDLPackTensor(DLManagedTensorUPtr tensor)
        : _capsule { enpycapsulate(tensor.get()) }
        , _device { tensor->dl_tensor.device }
    {
        tensor.release();  // the capsule now manages the tensor
    }

    // Non-owning view: the context-manager pool owns the tensor.
    explicit PyDLPackTensor(DLManagedTensor& tensor)
        : _capsule { enpycapsulate(&tensor, /* autodestruct = */ false) }
        , _device { tensor.dl_tensor.device }
    { }

    PyCapsule dlpack(pybind11::object = pybind11::none()) const
    {
        if (!PyCapsule_IsValid(_capsule.ptr(), kDLTensorCapsuleName))
            throw std::runtime_error("DLPack tensor has already been consumed.");
        return _capsule;
    }

    pybind11::tuple device() const
    {
        return pybind11::make_tuple(static_cast<int>(_device.device_type), _device.device_id);
    }

    const PyCapsule& capsule() const { return _capsule; }

private:
    static PyCapsule enpycapsulate(DLManagedTensor* tensor, bool autodestruct = true)
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

    PyCapsule _capsule;
    DLDevice _device;
};

// Manages the DLPack tensors created inside a SystemView context manager.
class DEFAULT_VISIBILITY DLPackTensorPool final {
public:
    // Takes a tensor and returns a non-owning view
    PyDLPackTensor manage(DLManagedTensorUPtr tensor)
    {
        auto view = PyDLPackTensor(*tensor);
        // Prevent a DLPack consumer from freeing
        // the tensor while the context manager is open.
        tensor->deleter = do_not_delete;
        _bundles.emplace_back(view.capsule(), std::move(tensor));
        return view;
    }

    // Invalidates all capsules, then frees the tensors.
    void clear()
    {
        while (!_bundles.empty()) {
            invalidate(_bundles.back());
            _bundles.pop_back();
        }
    }

private:
    using Bundle = std::tuple<PyCapsule, DLManagedTensorUPtr>;

    static void invalidate(Bundle& bundle)
    {
        auto obj = std::get<0>(bundle).ptr();

        if (PyCapsule_IsValid(obj, kDLTensorCapsuleName)) {
            PyCapsule_SetName(obj, kUsedDLTensorCapsuleName);
            PyCapsule_SetPointer(obj, opaque(&kInvalidDLManagedTensor));
        } else if (PyCapsule_IsValid(obj, kUsedDLTensorCapsuleName)) {
            PyCapsule_SetPointer(obj, opaque(&kInvalidDLManagedTensor));
        }
    }

    std::vector<Bundle> _bundles;
};

static DLPackTensorPool kTensorPool;

template <typename Property>
struct DEFAULT_VISIBILITY PyUnsafeEncapsulator final {
    static PyDLPackTensor wrap(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        return PyDLPackTensor(Property::from(sysview, location, mode));
    }
};

template <typename Property>
struct DEFAULT_VISIBILITY PyEncapsulator final {
    static PyDLPackTensor wrap(
        SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    )
    {
        if (!sysview.in_context_manager())
            throw std::runtime_error("Cannot access property outside a context manager.");
        return kTensorPool.manage(Property::from(sysview, location, mode));
    }
};

}  // namespace dlext
}  // namespace md
}  // namespace hoomd

#endif  // PY_HOOMD_DLPACK_EXTENSION_H_
