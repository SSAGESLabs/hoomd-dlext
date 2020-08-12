// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#include "PyDLSystemView.h"


using namespace dlext;

namespace py = pybind11;


void export_SystemView(py::module& m)
{
    py::class_<SystemView, std::shared_ptr<SystemView>>(m, "SystemView")
        .def(py::init<SystemDefinitionPtr>())
        .def("particle_data", &SystemView::particle_data)
        .def("is_gpu_enabled", &SystemView::is_gpu_enabled)
        .def("particle_number", &SystemView::particle_number)
        .def("get_device_id", &SystemView::get_device_id)
    ;
}

PYBIND11_MODULE(dlpack_extension, m)
{
    export_SystemView(m);

    py::enum_<AccessLocation>(m, "AccessLocation")
        .value("ON_HOST", kOnHost)
#ifdef ENABLE_CUDA
        .value("ON_DEVICE", kOnDevice)
#endif
    ;

    py::enum_<AccessMode>(m, "AccessMode")
        .value("READ", kRead)
        .value("READ_WRITE", kReadWrite)
        .value("OVERWRITE", kOverwrite)
    ;

    m.def("positions",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(positions(sysview, location, mode));
        }
    );
    m.def("types",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(types(sysview, location, mode));
        }
    );
    m.def("velocities",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(velocities(sysview, location, mode));
        }
    );
    m.def("masses",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(masses(sysview, location, mode));
        }
    );
    m.def("orientations",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(orientations(sysview, location, mode));
        }
    );
    m.def("angular_momenta",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(angular_momenta(sysview, location, mode));
        }
    );
    m.def("moments_of_intertia",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(moments_of_intertia(sysview, location, mode));
        }
    );
    m.def("charges",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(charges(sysview, location, mode));
        }
    );
    m.def("diameters",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(diameters(sysview, location, mode));
        }
    );
    m.def("images",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(images(sysview, location, mode));
        }
    );
    m.def("tags",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(tags(sysview, location, mode));
        }
    );
    m.def("net_forces",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(net_forces(sysview, location, mode));
        }
    );
    m.def("net_torques",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(net_torques(sysview, location, mode));
        }
    );
    m.def("net_virial",
        [](const SystemView& sysview, AccessLocation location, AccessMode mode) {
            return encapsulate(net_virial(sysview, location, mode));
        }
    );
}
