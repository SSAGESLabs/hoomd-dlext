// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef PY_HOOMD_DLPACK_EXTENSION_H_
#define PY_HOOMD_DLPACK_EXTENSION_H_


#include "DLExt.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"


namespace dlext
{


using PropertyExtractor =
    DLManagedTensorPtr (*)(const SystemView&, AccessLocation, AccessMode)
;

const char* const kDLTensorCapsuleName = "dltensor";


template <PropertyExtractor property>
inline pybind11::capsule encapsulate(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    auto dl_managed_tensor = property(sysview, location, mode);
    return pybind11::capsule(
        dl_managed_tensor, kDLTensorCapsuleName,
        [](PyObject* obj) {  // PyCapsule_Destructor
            auto dlmt = static_cast<DLManagedTensorPtr>(
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


#endif // PY_HOOMD_DLPACK_EXTENSION_H_
