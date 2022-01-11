// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#include "PyDLExt.h"
#include "PyHalfStepHook.h"
#include "Sampler.h"


using namespace dlext;
namespace py = pybind11;


void export_SystemView(py::module& m)
{
    py::class_<SystemView, std::shared_ptr<SystemView>>(m, "SystemView")
        .def(py::init<SystemDefinitionSPtr>())
        .def("particle_data", &SystemView::particle_data)
        .def("is_gpu_enabled", &SystemView::is_gpu_enabled)
        .def("local_particle_number", &SystemView::local_particle_number)
        .def("global_particle_number", &SystemView::global_particle_number)
        .def("get_device_id", &SystemView::get_device_id)
        .def("synchronize", &SystemView::synchronize)
    ;
}

void export_PyHalfStepHook(py::module m)
{
    using HalfStepHookSPtr = std::shared_ptr<HalfStepHook>;

    py::class_<HalfStepHook, PyHalfStepHook, HalfStepHookSPtr>(m, "HalfStepHook")
        .def(py::init<>())
        .def("setSystemDefinition", &HalfStepHook::setSystemDefinition)
        .def("update", &HalfStepHook::update)
    ;
}


PYBIND11_MODULE(dlpack_extension, m)
{
    // Enums
    py::enum_<AccessLocation>(m, "AccessLocation")
        .value("OnHost", kOnHost)
#ifdef ENABLE_CUDA
        .value("OnDevice", kOnDevice)
#endif
    ;

    py::enum_<AccessMode>(m, "AccessMode")
        .value("Read", kRead)
        .value("ReadWrite", kReadWrite)
        .value("Overwrite", kOverwrite)
    ;

    // Classes
    export_SystemView(m);
    export_PyHalfStepHook(m);
    export_Sampler(m);

    // Methods
    m.def("positions_types", encapsulate<&positions_types>);
    m.def("velocities_masses", encapsulate<&velocities_masses>);
    m.def("orientations", encapsulate<&orientations>);
    m.def("angular_momenta", encapsulate<&angular_momenta>);
    m.def("moments_of_intertia", encapsulate<&moments_of_intertia>);
    m.def("charges", encapsulate<&charges>);
    m.def("diameters", encapsulate<&diameters>);
    m.def("images", encapsulate<&images>);
    m.def("tags", encapsulate<&tags>);
    m.def("rtags", encapsulate<&rtags>);
    m.def("net_forces", encapsulate<&net_forces>);
    m.def("net_torques", encapsulate<&net_torques>);
    m.def("net_virial", encapsulate<&net_virial>);
}
