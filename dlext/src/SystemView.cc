// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#include "SystemView.h"
#include "cxx11utils.h"


using namespace dlext;
using namespace cxx11utils;


SystemView::SystemView(SystemDefinitionSPtr sysdef)
    : sysdef { sysdef }
    , pdata { sysdef->getParticleData() }
{
    exec_conf = pdata->getExecConf();
}

ParticleDataSPtr SystemView::particle_data() const { return pdata; }
ExecutionConfigurationSPtr SystemView::exec_config() const { return exec_conf; }
bool SystemView::is_gpu_enabled() const { return exec_conf->isCUDAEnabled(); }
unsigned int SystemView::local_particle_number() const { return pdata->getN(); }
unsigned int SystemView::global_particle_number() const { return pdata->getNGlobal(); }

int SystemView::get_device_id(bool gpu_flag) const
{
    maybe_unused(gpu_flag); // prevent compiler warnings when ENABLE_CUDA is not defined
#ifdef ENABLE_CUDA
    if (gpu_flag)
        return exec_conf->getGPUIds()[0];
#endif
    return exec_conf->getRank();
}

void SystemView::synchronize()
{
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled()) {
        auto gpu_ids = exec_conf->getGPUIds();
        for (int i = exec_conf->getNumActiveGPUs() - 1; i >= 0; --i) {
            cudaSetDevice(gpu_ids[i]);
            cudaDeviceSynchronize();
        }
    }
#endif
}
