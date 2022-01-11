// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef SAMPLER_H
#define SAMPLER_H

#include "hoomd/HalfStepHook.h"
#include "hoomd/GlobalArray.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include "dlpack/dlpack.h"

struct DLDataBridge {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DLManagedTensor tensor;
};


class Sampler : public HalfStepHook
    {
    public:
        //! Constructor
      Sampler(std::shared_ptr<SystemDefinition> sysdef, pybind11::function python_update);

      virtual void setSystemDefinition(std::shared_ptr<SystemDefinition> sysdef) override;

        //! Take one timestep forward
      virtual void update(unsigned int timestep) override;

      // run a custom python function on data from hoomd
      // access_mode is ignored for forces. Forces are returned in readwrite mode always.
      void run_on_data(pybind11::function py_exec, const access_location::Enum location, const access_mode::Enum mode);

    private:
      template<typename TS, typename TV>
      DLDataBridge wrap(TS* const ptr, const bool, const int64_t size2 = 1, const uint64_t offset=0, uint64_t stride1_offset = 0);
      pybind11::function m_python_update;
      std::shared_ptr<SystemDefinition> m_sysdef;
      std::shared_ptr<ParticleData> m_pdata;
      std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    };

void export_Sampler(pybind11::module& m);

#endif//SAMPLER_H
