// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef HOOMD_DLPACK_EXTENSION_H_
#define HOOMD_DLPACK_EXTENSION_H_

#include <vector>

#include "SystemView.h"
#include "dlpack/dlpack.h"


namespace dlext
{


using DLManagedTensorPtr = DLManagedTensor*;

using AccessLocation = access_location::Enum;
const auto kOnHost = access_location::host;
#ifdef ENABLE_CUDA
const auto kOnDevice = access_location::device;
#endif

using AccessMode = access_mode::Enum;
const auto kRead = access_mode::read;
const auto kReadWrite = access_mode::readwrite;
const auto kOverwrite = access_mode::overwrite;

constexpr uint8_t kBits = std::is_same<Scalar, float>::value ? 32 : 64;

template <template <typename> class Array, typename T, typename Object>
using PropertyGetter = const Array<T>& (Object::*)() const;

template <typename T>
using ArrayHandleUPtr = std::unique_ptr<ArrayHandle<T>>;

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
using DLDataBridgeUPtr = std::unique_ptr<DLDataBridge<T>>;

template <typename T>
void DLDataBridgeDeleter(DLManagedTensorPtr tensor)
{
    if (tensor)
        delete static_cast<DLDataBridge<T>*>(tensor->manager_ctx);
}

template <typename T>
inline void* opaque(T* data) { return static_cast<void*>(data); }

inline DLDevice dldevice(const SystemView& sysview, bool gpu_flag)
{
    return DLDevice { gpu_flag ? kDLCUDA : kDLCPU, sysview.get_device_id(gpu_flag) };
}

template <typename>
constexpr DLDataType dtype();
template <>
constexpr DLDataType dtype<Scalar4>() { return DLDataType {kDLFloat, kBits, 1}; }
template <>
constexpr DLDataType dtype<Scalar3>() { return DLDataType {kDLFloat, kBits, 1}; }
template <>
constexpr DLDataType dtype<Scalar>() { return DLDataType {kDLFloat, kBits, 1}; }
template <>
constexpr DLDataType dtype<int3>() { return DLDataType {kDLInt, 32, 1}; }
template <>
constexpr DLDataType dtype<unsigned int>() { return DLDataType {kDLUInt, 32, 1}; }

template <template <typename> class>
unsigned int particle_number(const SystemView& sysview);
template <>
inline unsigned int particle_number<GlobalArray>(const SystemView& sysview)
{
    return sysview.local_particle_number();
}
template <>
inline unsigned int particle_number<GlobalVector>(const SystemView& sysview)
{
    return sysview.global_particle_number();
}

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
DLManagedTensorPtr wrap(
    const SystemView& sysview, PropertyGetter<A, T, O> getter,
    AccessLocation requested_location, AccessMode mode,
    int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0
) {
    assert((size2 >= 1)); // assert is a macro so the extra parentheses are requiered here

    auto location = sysview.is_gpu_enabled() ? requested_location : kOnHost;
    auto handle = ArrayHandleUPtr<T>(
        new ArrayHandle<T>(INVOKE(*(sysview.particle_data()), getter)(), location, mode)
    );
    auto bridge = DLDataBridgeUPtr<T>(new DLDataBridge<T>(handle));

#ifdef ENABLE_CUDA
    auto gpu_flag = (location == kOnDevice);
#else
    auto gpu_flag = false;
#endif

    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = DLDataBridgeDeleter<T>;

    auto& dltensor = bridge->tensor.dl_tensor;
    dltensor.data = opaque(bridge->handle->data);
    dltensor.device = dldevice(sysview, gpu_flag);
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

inline DLManagedTensorPtr positions_types(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getPositions, location, mode, 4);
}

inline DLManagedTensorPtr velocities_masses(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getVelocities, location, mode, 4);
}

inline DLManagedTensorPtr orientations(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getOrientationArray, location, mode, 4);
}

inline DLManagedTensorPtr angular_momenta(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getAngularMomentumArray, location, mode, 4);
}

inline DLManagedTensorPtr moments_of_intertia(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getMomentsOfInertiaArray, location, mode, 3);
}

inline DLManagedTensorPtr charges(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getCharges, location, mode);
}

inline DLManagedTensorPtr diameters(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getDiameters, location, mode);
}

inline DLManagedTensorPtr images(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getImages, location, mode, 3);
}

inline DLManagedTensorPtr tags(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getTags, location, mode);
}

inline DLManagedTensorPtr rtags(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getRTags, location, mode);
}

inline DLManagedTensorPtr net_forces(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getNetForce, location, mode, 4);
}

inline DLManagedTensorPtr net_torques(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getNetTorqueArray, location, mode, 4);
}

inline DLManagedTensorPtr net_virial(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ParticleData::getNetVirial, location, mode, 6);
}


}  // namespace dlext


#endif  // HOOMD_DLPACK_EXTENSION_H_
