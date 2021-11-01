// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef SAMPLER_H
#define SAMPLER_H

#include "hoomd/HalfStepHook.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include "dlpack.h"


class Sampler : public HalfStepHook
    {
    public:
        //! Constructor
      Sampler(std::shared_ptr<SystemDefinition> sysdef,
              pybind11::function python_update);

      virtual void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef);

        //! Take one timestep forward
      virtual void update(unsigned int timestep);
    protected:
      template<typename T>
      DLTensor wrap(T* const ptr, const int64_t size2 = 1, const uint64_t offset=0, uint64_t stride1_offset = 0);
      pybind11::function m_python_update;
      std::shared_ptr<SystemDefinition> m_sysdef;
      std::shared_ptr<ParticleData> m_pdata;
      std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    };

void export_Sampler(pybind11::module& m);

#endif//SAMPLER_H
