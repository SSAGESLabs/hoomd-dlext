// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef HOOMD_SYSVIEW_H_
#define HOOMD_SYSVIEW_H_


#include "cxx11utils.h"
#include "DLExt.h"

#include "hoomd/SystemDefinition.h"

#include <unordered_set>


namespace dlext
{

namespace cxx11 = cxx11utils;


class SystemView;


// { // Aliases

using AccessLocation = access_location::Enum;
const auto kOnHost = access_location::host;
#ifdef ENABLE_CUDA
const auto kOnDevice = access_location::device;
#endif

using AccessMode = access_mode::Enum;
const auto kRead = access_mode::read;
const auto kReadWrite = access_mode::readwrite;
const auto kOverwrite = access_mode::overwrite;

using ParticleDataSPtr = std::shared_ptr<ParticleData>;
using SystemDefinitionSPtr = std::shared_ptr<SystemDefinition>;
using ExecutionConfigurationSPtr = std::shared_ptr<const ExecutionConfiguration>;

template <template <typename> class Array, typename T, typename Object>
using ArrayPropertyGetter = const Array<T>& (Object::*)() const;

template <typename T>
using PropertyGetter = T (*)(const SystemView&, AccessLocation, AccessMode);

// } // Aliases

class DEFAULT_VISIBILITY SystemView {
public:
    SystemView(SystemDefinitionSPtr sysdef);
    ParticleDataSPtr particle_data() const;
    ExecutionConfigurationSPtr exec_config() const;
    bool is_gpu_enabled() const;
    bool in_context_manager() const;
    unsigned int local_particle_number() const;
    unsigned int global_particle_number() const;
    int get_device_id(bool gpu_flag) const;
    void synchronize();
    void enter();
    void exit();
private:
    SystemDefinitionSPtr _sysdef;
    ParticleDataSPtr _pdata;
    ExecutionConfigurationSPtr _exec_conf;
    bool _in_context_manager = false;
};


inline DLDevice dldevice(const SystemView& sysview, bool gpu_flag)
{
    return DLDevice { gpu_flag ? kDLCUDA : kDLCPU, sysview.get_device_id(gpu_flag) };
}

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

template <template <typename> class A, typename T, typename O>
DLManagedTensorPtr wrap(
    const SystemView& sysview, ArrayPropertyGetter<A, T, O> getter,
    AccessLocation requested_location, AccessMode mode,
    int64_t size2 = 1, uint64_t offset = 0, uint64_t stride1_offset = 0
) {
    assert((size2 >= 1));

    auto location = sysview.is_gpu_enabled() ? requested_location : kOnHost;
    auto handle = cxx11utils::make_unique<ArrayHandle<T>>(
        INVOKE(*(sysview.particle_data()), getter)(), location, mode
    );
    auto bridge = cxx11utils::make_unique<DLDataBridge<T>>(handle);

#ifdef ENABLE_CUDA
    auto gpu_flag = (location == kOnDevice);
#else
    auto gpu_flag = false;
#endif

    bridge->tensor.manager_ctx = bridge.get();
    bridge->tensor.deleter = delete_bridge<T>;

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


struct PositionsTypes {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getPositions, location, mode, 4);
    }
};

struct VelocitiesMasses {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getVelocities, location, mode, 4);
    }
};

struct Orientations {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getOrientationArray, location, mode, 4);
    }
};

struct AngularMomenta {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getAngularMomentumArray, location, mode, 4);
    }
};

struct MomentsOfInertia {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getMomentsOfInertiaArray, location, mode, 3);
    }
};

struct Charges {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getCharges, location, mode);
    }
};

struct Diameters {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getDiameters, location, mode);
    }
};

struct Images {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getImages, location, mode, 3);
    }
};

struct Tags {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getTags, location, mode);
    }
};

struct RTags {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getRTags, location, mode);
    }
};

struct NetForces {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getNetForce, location, mode, 4);
    }
};

struct NetTorques{
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getNetTorqueArray, location, mode, 4);
    }
};

struct NetVirial {
    static DLManagedTensorPtr from(
        const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
    ) {
        return wrap(sysview, &ParticleData::getNetVirial, location, mode, 6);
    }
};


}  // namespace dlext


#endif  // HOOMD_SYSVIEW_H_
