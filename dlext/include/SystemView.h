// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef HOOMD_SYSVIEW_H_
#define HOOMD_SYSVIEW_H_


#include <memory>
#include <type_traits>

#include "cxx11utils.h"

#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/SystemDefinition.h"


namespace dlext
{


using ParticleDataSPtr = std::shared_ptr<ParticleData>;
using SystemDefinitionSPtr = std::shared_ptr<SystemDefinition>;
using ExecutionConfigurationSPtr = std::shared_ptr<const ExecutionConfiguration>;


class DEFAULT_VISIBILITY SystemView {
public:
    SystemView(SystemDefinitionSPtr sysdef);
    ParticleDataSPtr particle_data() const;
    ExecutionConfigurationSPtr exec_config() const;
    bool is_gpu_enabled() const;
    unsigned int local_particle_number() const;
    unsigned int global_particle_number() const;
    int get_device_id(bool gpu_flag) const;
    void synchronize();
private:
    SystemDefinitionSPtr sysdef;
    ParticleDataSPtr pdata;
    ExecutionConfigurationSPtr exec_conf;
};


}  // namespace dlext


#endif  // HOOMD_SYSVIEW_H_
