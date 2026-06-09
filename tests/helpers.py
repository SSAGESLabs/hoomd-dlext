# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md

"""Shared utilities for the CPU tests."""

import ctypes

import numpy as np

import hoomd
from hoomd.dlext import AccessLocation

N = 8
LOC = AccessLocation.OnHost


def read_tensor(tensor):
    """Return a copy of a tensor's data."""
    return np.array(np.from_dlpack(tensor))


def writable_tensor(tensor):
    """Return a writable NumPy view over a tensor's memory."""
    array = np.from_dlpack(tensor)
    address = array.__array_interface__["data"][0]
    pointer = ctypes.cast(
        address, ctypes.POINTER(np.ctypeslib.as_ctypes_type(array.dtype))
    )
    return np.ctypeslib.as_array(pointer, shape=array.shape)


def _nve_method():
    """Backward compatible NVE integration method."""
    methods = hoomd.md.methods
    cls = getattr(methods, "ConstantVolume", None) or methods.NVE
    return cls(filter=hoomd.filter.All())


def _unsorted_positions():
    """Positions chosen so ParticleSorter changes order."""
    xs = np.linspace(-8.0, 8.0, N)
    return np.column_stack([-xs, np.zeros(N), np.zeros(N)])


def _snapshot(positions):
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.configuration.box = [20, 20, 20, 0, 0, 0]
    snap.particles.position[:] = positions
    snap.particles.typeid[:] = 0
    snap.particles.types = ["A"]
    # Encode each particle's tag in velocity.x so identity survives reorders.
    snap.particles.velocity[:] = np.column_stack(
        [np.arange(N, dtype=float), np.zeros(N), np.zeros(N)]
    )
    return snap


def generate_simulation(*, sort=False):
    """Build a CPU simulation whose particle identity survives reordering."""
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(_snapshot(_unsorted_positions()))
    if sort:
        sim.operations.tuners.append(
            hoomd.tune.ParticleSorter(trigger=hoomd.trigger.Periodic(1))
        )
    integrator = hoomd.md.Integrator(dt=0.0)
    integrator.methods.append(_nve_method())
    sim.operations.integrator = integrator
    return sim
