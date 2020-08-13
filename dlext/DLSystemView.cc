// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#include "DLSystemView.h"


using namespace dlext;
using namespace utils;


SystemView::SystemView(SystemDefinitionPtr sysdef)
    : sysdef { sysdef }
    , pdata { sysdef->getParticleData() }
{
    exec_conf = pdata->getExecConf();
}

ParticleDataPtr SystemView::particle_data() const { return pdata; }
ExecutionConfigurationPtr SystemView::exec_config() const { return exec_conf; }
bool SystemView::is_gpu_enabled() const { return exec_conf->isCUDAEnabled(); }
unsigned int SystemView::particle_number() const { return pdata->getN(); }

int SystemView::get_device_id(bool gpu_flag) const {
    maybe_unused(gpu_flag); // prevent compiler warnings when ENABLE_CUDA is not defined
#ifdef ENABLE_CUDA
    if (gpu_flag)
        return exec_conf->getGPUIds()[0];
#endif
    return exec_conf->getRank();
}


/*
DLManagedTensorPtr forces(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ForceCompute::getForce, location, mode, 4);
}
DLManagedTensorPtr torques(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ForceCompute::getTorque, location, mode, 4);
}
DLManagedTensorPtr virial(
    const SystemView& sysview, AccessLocation location, AccessMode mode = kReadWrite
) {
    return wrap(sysview, &ForceCompute::getVirial, location, mode, 6, 0, 5);
}
*/
