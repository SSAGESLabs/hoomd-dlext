// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef PY_HOOMD_HS_HOOK_H_
#define PY_HOOMD_HS_HOOK_H_


#include "SystemView.h"

#include "hoomd/HalfStepHook.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"


namespace dlext
{


//! Trampoline class to allow overriding HalfStepHook mathods from within Pyhton.
//!
//! References:
//! - https://pybind11.readthedocs.io/en/stable/advanced/classes.html
//!
class PyHalfStepHook : public HalfStepHook {
public:
    using HalfStepHook::HalfStepHook;

    void setSystemDefinition(SystemDefinitionSPtr sysdef) override
    {
        PYBIND11_OVERLOAD_PURE(void, HalfStepHook, setSystemDefinition, sysdef);
    }

    void update(unsigned int timestep) override
    {
        PYBIND11_OVERLOAD_PURE(void, HalfStepHook, update, timestep);
    }
};


} // namespace dlext


#endif // PY_HOOMD_HS_HOOK_H_
