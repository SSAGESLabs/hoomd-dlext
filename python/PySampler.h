// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef PY_SAMPLER_H_
#define PY_SAMPLER_H_


#include "Sampler.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"


namespace dlext
{


using PyFunction = pybind11::function;


class DEFAULT_VISIBILITY PySampler final : public Sampler<PyFunction, PyUnsafeEncapsulator> {
public:
    using Sampler<PyFunction, PyUnsafeEncapsulator>::Sampler;
};


}  // namespace dlext


#endif  // PY_SAMPLER_H_
