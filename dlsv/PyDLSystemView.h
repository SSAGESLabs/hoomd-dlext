// SPDX-License-Identifier: MIT
// This file is part of `HOOMD-dlpack`, see LICENSE.md

#pragma once

#include "DLSystemView.h"

#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#endif


void export_SystemView(pybind11::module& m, std::string name)
{
    using namespace dlsv;
    pybind11::class_< SystemView, std::shared_ptr<SystemView> >(m, name.c_str())
        .def(pybind11::init<SystemDefinitionPtr>())
        .def("particle_data", &SystemView::particle_data)
        .def("exec_conf", &SystemView::exec_conf)
        .def("is_gpu_enabled", &SystemView::is_gpu_enabled)
        .def("particle_number", &SystemView::particle_number)
        .def("get_device_id", &SystemView::get_device_id)
    ;
}
