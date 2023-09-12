// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef HOOMD_DLPACK_EXTENSION_H_
#define HOOMD_DLPACK_EXTENSION_H_

#include "SystemView.h"
#include "dlpack/dlpack.h"

#include <type_traits>
#include <vector>

namespace hoomd
{
namespace md
{
namespace dlext
{

namespace cxx11 = cxx11utils;

// { // Aliases

using DLManagedTensorDeleter = void (*)(DLManagedTensor*);

template <typename T>
using ArrayHandleUPtr = std::unique_ptr<ArrayHandle<T>>;

template <template <typename> class Array, typename T, typename Object>
using ArrayPropertyGetter = const Array<T>& (Object::*)() const;

template <typename T>
using PropertyGetter = T (*)(const SystemView&, AccessLocation, AccessMode);

// } // Aliases

// { // Constants

constexpr uint8_t kBits = std::is_same<Scalar, float>::value ? 32 : 64;

constexpr DLManagedTensor kInvalidDLManagedTensor {
    DLTensor {
        nullptr,  // data
        DLDevice { kDLExtDev, -1 },  // device
        -1,  // ndim
        DLDataType { 0, 0, 0 },  // dtype
        nullptr,  // shape
        nullptr,  // stride
        0  // byte_offset
    },
    nullptr,
    nullptr
};

// } // Constants

template <typename T>
struct DLDataBridge {
    ArrayHandleUPtr<T> handle;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;

    DLDataBridge(ArrayHandleUPtr<T>& handle)
        : handle { std::move(handle) }
    { }
};

template <typename T>
void delete_bridge(DLManagedTensor* tensor)
{
    if (tensor)
        delete static_cast<DLDataBridge<T>*>(tensor->manager_ctx);
}

void do_not_delete(DLManagedTensor* tensor) { }

template <typename T>
inline void* opaque(T* data) { return static_cast<void*>(data); }

template <typename T>
inline void* opaque(const T* data) { return (void*)(data); }

inline DLDevice device_info(const SystemView& sysview, AccessLocation location)
{
#ifdef ENABLE_CUDA
    auto gpu_flag = (location == kOnDevice);
#else
    auto gpu_flag = false;
#endif
    return DLDevice { gpu_flag ? kDLCUDA : kDLCPU, sysview.get_device_id(gpu_flag) };
}

template <typename>
constexpr DLDataType dtype();
template <>
constexpr DLDataType dtype<Scalar4>() { return DLDataType { kDLFloat, kBits, 1 }; }
template <>
constexpr DLDataType dtype<Scalar3>() { return DLDataType { kDLFloat, kBits, 1 }; }
template <>
constexpr DLDataType dtype<Scalar>() { return DLDataType { kDLFloat, kBits, 1 }; }
template <>
constexpr DLDataType dtype<int3>() { return DLDataType { kDLInt, 32, 1 }; }
template <>
constexpr DLDataType dtype<unsigned int>() { return DLDataType { kDLUInt, 32, 1 }; }

template <typename>
constexpr int64_t stride1();
template <>
constexpr int64_t stride1<Scalar4>() { return 4; }
template <>
constexpr int64_t stride1<Scalar3>() { return 3; }
template <>
constexpr int64_t stride1<Scalar>() { return 1; }
template <>
constexpr int64_t stride1<int3>() { return 3; }
template <>
constexpr int64_t stride1<unsigned int>() { return 1; }

template <template <typename> class A, typename T, typename O>
DLManagedTensor* wrap(
    const SystemView& sysview, ArrayPropertyGetter<A, T, O> getter,
    AccessLocation requested_location, AccessMode mode,
    int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0
)
{
    assert((size2 >= 1));

    auto location = sysview.is_gpu_enabled() ? requested_location : kOnHost;
    auto handle = cxx11::make_unique<ArrayHandle<T>>(
        INVOKE(*(sysview.particle_data()), getter)(), location, mode
    );
    auto bridge = cxx11::make_unique<DLDataBridge<T>>(handle);

    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = delete_bridge<T>;

    auto& dltensor = bridge->tensor.dl_tensor;
    dltensor.data = opaque(bridge->handle->data);
    dltensor.device = device_info(sysview, location);
    dltensor.dtype = dtype<T>();

    auto& shape = bridge->shape;
    shape.push_back(particle_number<A>(sysview));
    if (size2 > 1)
        shape.push_back(size2);

    auto& strides = bridge->strides;
    strides.push_back(stride1<T>() + stride1_offset);
    if (size2 > 1)
        strides.push_back(1);

    dltensor.ndim = shape.size();
    dltensor.shape = reinterpret_cast<std::int64_t*>(shape.data());
    dltensor.strides = reinterpret_cast<std::int64_t*>(strides.data());
    dltensor.byte_offset = offset;

    return &(bridge.release()->tensor);
}

#define DLEXT_PROPERTY_WRAPPER(PROPERTY, GETTER, SIZE1)                                      \
    struct PROPERTY final {                                                                  \
        static DLManagedTensor* from(                                                        \
            const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite \
        )                                                                                    \
        {                                                                                    \
            return wrap(sysview, GETTER, location, mode, SIZE1);                             \
        }                                                                                    \
    };

DLEXT_PROPERTY_WRAPPER(PositionsTypes, &ParticleData::getPositions, 4)
DLEXT_PROPERTY_WRAPPER(VelocitiesMasses, &ParticleData::getVelocities, 4)
DLEXT_PROPERTY_WRAPPER(Orientations, &ParticleData::getOrientationArray, 4)
DLEXT_PROPERTY_WRAPPER(AngularMomenta, &ParticleData::getAngularMomentumArray, 4)
DLEXT_PROPERTY_WRAPPER(MomentsOfInertia, &ParticleData::getMomentsOfInertiaArray, 3)
DLEXT_PROPERTY_WRAPPER(Charges, &ParticleData::getCharges, 1)
DLEXT_PROPERTY_WRAPPER(Diameters, &ParticleData::getDiameters, 1)
DLEXT_PROPERTY_WRAPPER(Images, &ParticleData::getImages, 3)
DLEXT_PROPERTY_WRAPPER(Tags, &ParticleData::getTags, 1)
DLEXT_PROPERTY_WRAPPER(RTags, &ParticleData::getRTags, 1)
DLEXT_PROPERTY_WRAPPER(NetForces, &ParticleData::getNetForce, 4)
DLEXT_PROPERTY_WRAPPER(NetTorques, &ParticleData::getNetTorqueArray, 4)
DLEXT_PROPERTY_WRAPPER(NetVirial, &ParticleData::getNetVirial, 6)

#undef DLEXT_PROPERTY_WRAPPER

}  // namespace dlext
}  // namespace md
}  // namespace hoomd

#endif  // HOOMD_DLPACK_EXTENSION_H_
