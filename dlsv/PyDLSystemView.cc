// SPDX-License-Identifier: MIT
// This file is part of `HOOMD-dlpack`, see LICENSE.md

#include "PyDLSystemView.h"


using namespace dlsv


PYBIND11_MODULE(dlsv, m) {
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
