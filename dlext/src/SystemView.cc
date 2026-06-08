// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#include "SystemView.h"

using namespace hoomd::md::dlext;

SystemView::SystemView(SPtr<System> system)
    : _system { system }
    , _pdata { system->getSystemDefinition()->getParticleData() }
{
    _exec_conf = _pdata->getExecConf();
}

SPtr<System> SystemView::system() { return _system; }
SPtr<ParticleData> SystemView::particle_data() const { return _pdata; }
SPtr<const ExecutionConfiguration> SystemView::exec_config() const { return _exec_conf; }
bool SystemView::is_gpu_enabled() const { return _exec_conf->isCUDAEnabled(); }
bool SystemView::in_context_manager() const { return _in_context_manager; }
unsigned int SystemView::local_particle_number() const { return _pdata->getN(); }
unsigned int SystemView::global_particle_number() const { return _pdata->getNGlobal(); }

int SystemView::get_device_id(bool gpu_flag) const
{
#ifdef ENABLE_CUDA
    if (gpu_flag) {
#ifdef HOOMD5
        return _exec_conf->getGPUId();
#else
        return _exec_conf->getGPUIds()[0];
#endif
    }
#else
    maybe_unused(gpu_flag);
#endif
    return 0;
}

void SystemView::synchronize()
{
#ifdef ENABLE_CUDA
    if (_exec_conf->isCUDAEnabled()) {
#ifdef HOOMD5
        cudaSetDevice(_exec_conf->getGPUId());
        cudaDeviceSynchronize();
#else
        auto gpu_ids = _exec_conf->getGPUIds();
        for (int i = _exec_conf->getNumActiveGPUs() - 1; i >= 0; --i) {
            cudaSetDevice(gpu_ids[i]);
            cudaDeviceSynchronize();
        }
#endif
    }
#endif
}

void SystemView::enter()
{
    if (_in_context_manager)
        throw std::runtime_error("Context manager scope already active.");
    _in_context_manager = true;
}

void SystemView::exit()
{
    _in_context_manager = false;
}
