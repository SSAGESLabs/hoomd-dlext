# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md


from .dlpack_extension import *


if hasattr(AccessLocation, 'OnDevice'):
    DEFAULT_DEVICE = AccessLocation.OnDevice
else:
    DEFAULT_DEVICE = AccessLocation.OnHost


def view(context):
    dt = context.integrator.dt
    system_view = SystemView(context.system_definition)
    #
    positions = positions_types(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite)
    vel_mass = velocities_masses(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite)
    forces = net_forces(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite)
    ids = tags(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite)
    #
    box = system_view.particle_data().getGlobalBox()
    L  = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = (
        (L.x, xy * L.y, xz * L.z, 0.0),  # Last column is a hack until
        (0.0,      L.y, yz * L.z, 0.0),  # https://github.com/google/jax/issues/4196
        (0.0,      0.0,      L.z, 0.0)   # gets fixed
    )
    origin = (lo.x, lo.y, lo.z)
    #
    return (positions, vel_mass, forces, ids, (H, origin), dt)


class Hook(HalfStepHook):
    def initialize_from(self, sampler):
        initialize, updater = sampler
        self.state = initialize()
        self.update_from = updater
        return None
    #
    def update(self, timestep):
        self.state = self.update_from(self.state, timestep)
        return None


def attach(context, hook):
    context.integrator.cpp_integrator.setHalfStepHook(hook)
    return None
