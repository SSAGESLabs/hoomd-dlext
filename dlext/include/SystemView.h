// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef HOOMD_SYSVIEW_H_
#define HOOMD_SYSVIEW_H_

#include "cxx11utils.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/System.h"

namespace hoomd
{
namespace md
{
namespace dlext
{

using namespace cxx11utils;
using namespace hoomd;

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

// { // Aliases

class DEFAULT_VISIBILITY SystemView {
public:
    SystemView(SPtr<System> system);
    SPtr<System> system();
    SPtr<ParticleData> particle_data() const;
    SPtr<const ExecutionConfiguration> exec_config() const;
    bool is_gpu_enabled() const;
    bool in_context_manager() const;
    unsigned int local_particle_number() const;
    unsigned int global_particle_number() const;
    int get_device_id(bool gpu_flag) const;
    void synchronize();
    void enter();
    void exit();

private:
    SPtr<System> _system;
    SPtr<ParticleData> _pdata;
    SPtr<const ExecutionConfiguration> _exec_conf;
    bool _in_context_manager = false;
};

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

}  // namespace dlext
}  // namespace md
}  // namespace hoomd

#endif  // HOOMD_SYSVIEW_H_
