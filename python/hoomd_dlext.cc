// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#include "PyDLExt.h"
#ifdef EXPORT_HALFSTEPHOOK
#include "PyHalfStepHook.h"
#endif
#include "Sampler.h"

namespace py = pybind11;
using namespace hoomd::md::dlext;

void export_SystemView(py::module& m)
{
    using PyObject = py::object;

    py::class_<SystemView>(m, "SystemView")
        .def(py::init<SPtr<System>>())
        .def_property_readonly("system", &SystemView::system)
        .def_property_readonly("particle_data", &SystemView::particle_data)
        .def_property_readonly("is_gpu_enabled", &SystemView::is_gpu_enabled)
        .def_property_readonly("local_particle_number", &SystemView::local_particle_number)
        .def_property_readonly("global_particle_number", &SystemView::global_particle_number)
        .def("synchronize", &SystemView::synchronize)
        .def("__enter__", [](SystemView& self) { self.enter(); return self; })
        .def("__exit__", [](SystemView& self, PyObject, PyObject, PyObject) {
            while (kPyCapsulesPool.size() > 0) {
                invalidate(kPyCapsulesPool.back());
                kPyCapsulesPool.pop_back();
            }
            self.exit();
        });
}

void export_PySampler(py::module m)
{
    using PyFunction = py::function;
    using PySampler = Sampler<PyFunction, PyUnsafeEncapsulator>;

#ifdef EXPORT_HALFSTEPHOOK
    py::class_<HalfStepHook, PyHalfStepHook, SPtr<HalfStepHook>>(m, "HalfStepHook")
        .def(py::init<>())
        .def("update", &HalfStepHook::update);
#endif

    py::class_<PySampler, SPtr<PySampler>, HalfStepHook>(m, "DLExtSampler")
        .def(py::init<SystemView&, PyFunction, AccessLocation, AccessMode>())
        .def("system_view", &PySampler::system_view)
        .def("forward_data", &PySampler::forward_data<PyFunction>)
        .def("update", &PySampler::update);
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
        .value("Overwrite", kOverwrite);

    // Classes
    export_SystemView(m);
    export_PySampler(m);

    // Methods
    m.def("positions_types", &PyEncapsulator<PositionsTypes>::wrap);
    m.def("velocities_masses", &PyEncapsulator<VelocitiesMasses>::wrap);
    m.def("orientations", &PyEncapsulator<Orientations>::wrap);
    m.def("angular_momenta", &PyEncapsulator<AngularMomenta>::wrap);
    m.def("moments_of_intertia", &PyEncapsulator<MomentsOfInertia>::wrap);
    m.def("charges", &PyEncapsulator<Charges>::wrap);
    m.def("diameters", &PyEncapsulator<Diameters>::wrap);
    m.def("images", &PyEncapsulator<Images>::wrap);
    m.def("tags", &PyEncapsulator<Tags>::wrap);
    m.def("rtags", &PyEncapsulator<RTags>::wrap);
    m.def("net_forces", &PyEncapsulator<NetForces>::wrap);
    m.def("net_torques", &PyEncapsulator<NetTorques>::wrap);
    m.def("net_virial", &PyEncapsulator<NetVirial>::wrap);
}
