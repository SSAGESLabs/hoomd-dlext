// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef PY_HOOMD_HS_HOOK_H_
#define PY_HOOMD_HS_HOOK_H_

#include "hoomd/HalfStepHook.h"
#ifdef HOOMD2
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#else
#include <pybind11/pybind11.h>
#endif
#include "CallbackHandler.h"

namespace hoomd
{
namespace md
{
namespace dlext
{

//! Trampoline class to allow overriding HalfStepHook mathods from within Pyhton.
//!
//! References:
//! - https://pybind11.readthedocs.io/en/stable/advanced/classes.html
//!
class DEFAULT_VISIBILITY PyHalfStepHook : public HalfStepHook {
public:
    using HalfStepHook::HalfStepHook;

    void setSystemDefinition(SPtr<SystemDefinition> sysdef) override
    {
        PYBIND11_OVERLOAD_PURE(void, HalfStepHook, setSystemDefinition, sysdef);
    }

    void update(TimeStep timestep) override
    {
        PYBIND11_OVERLOAD_PURE(void, HalfStepHook, update, timestep);
    }
};

}  // namespace dlext
}  // namespace md
}  // namespace hoomd

#endif  // PY_HOOMD_HS_HOOK_H_
