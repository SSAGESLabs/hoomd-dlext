// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#include "DLSystemView.h"

#ifndef NVCC
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#endif


using namespace dlext


void export_SystemView(pybind11::module& m, std::string name)
{
    pybind11::class_<SystemView, std::shared_ptr<SystemView>> sysview(m, name.c_str());
    sysview
        .def(pybind11::init<SystemDefinitionPtr>())
        .def("particle_data", &SystemView::particle_data)
        .def("exec_conf", &SystemView::exec_conf)
        .def("is_gpu_enabled", &SystemView::is_gpu_enabled)
        .def("particle_number", &SystemView::particle_number)
        .def("get_device_id", &SystemView::get_device_id)
    ;
}

PYBIND11_MODULE(dlpack_extension, m) {
    export_SystemView(m, "SystemView");

    m.def("positions", &positions);
    m.def("types", &types);
    m.def("velocities", &velocities);
    m.def("masses", &masses);
    m.def("orientations", &orientations);
    m.def("angular_momenta", &angular_momenta);
    m.def("moments_of_intertia", &moments_of_intertia);
    m.def("charges", &charges);
    m.def("diameters", &diameters);
    m.def("images", &images);
    m.def("tags", &tags);
    m.def("net_forces", &net_forces);
    m.def("net_torques", &net_torques);
    m.def("net_virial", &net_virial);
}
