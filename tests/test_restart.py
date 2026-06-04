# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md

"""
These tests force a deterministic CPU particle reorder, then check the two pieces
needed for a correct restart:

- ``dlext.tags`` reports the current tag map and is the inverse of
  callback ``rtags``.
- A one-time, tag-ordered restore through ``AccessMode.Overwrite`` writes saved
  ordered data into hoomd's current order.
"""

import ctypes

import numpy as np
import pytest

hoomd = pytest.importorskip("hoomd")
dlext = pytest.importorskip("hoomd.dlext")

from hoomd.dlext import (  # noqa: E402
    AccessLocation,
    AccessMode,
    DLExtSampler,
    SystemView,
)

N = 8
LOC = AccessLocation.OnHost


class _DLPackCapsule:
    """Adapt a raw DLPack ``dltensor`` capsule for numpy."""

    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, *args, **kwargs):
        return self._capsule

    def __dlpack_device__(self):
        return (1, 0)  # kDLCPU, device 0


def _read(capsule):
    """A copy of the capsule's data (numpy imports DLPack arrays read-only)."""
    return np.array(np.from_dlpack(_DLPackCapsule(capsule)))


def _writable(capsule):
    """A writable view over the capsule's memory."""
    array = np.from_dlpack(_DLPackCapsule(capsule))
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


def _simulation(positions, *, sort):
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(_snapshot(positions))
    if sort:
        sim.operations.tuners.append(
            hoomd.tune.ParticleSorter(trigger=hoomd.trigger.Periodic(1))
        )
    integrator = hoomd.md.Integrator(dt=0.0)
    integrator.methods.append(_nve_method())
    sim.operations.integrator = integrator
    return sim


def _tags(view):
    with view:
        return _read(dlext.tags(view, LOC, AccessMode.Read)).astype(int)


def _forward(view, mode, callback):
    DLExtSampler(view, lambda *args: None, LOC, AccessMode.Read).forward_data(
        callback, LOC, mode, 0
    )


def _read_bundle(view):
    out = {}

    def callback(_positions, vel_mass, rtags, *_):
        out["vx"] = _read(vel_mass)[:, 0]
        out["rtags"] = _read(rtags).astype(int)

    _forward(view, AccessMode.Read, callback)
    return out


def test_tags_accessor_inverts_rtags_after_reorder():
    """``dlext.tags`` is reorder-aware and the inverse of ``rtags``."""
    sim = _simulation(_unsorted_positions(), sort=True)
    sim.run(2)
    view = SystemView(sim._cpp_sys)

    tags = _tags(view)
    assert not np.array_equal(
        tags, np.arange(N)
    ), "ParticleSorter did not change the particle order"

    rtags = _read_bundle(view)["rtags"]
    # tags and rtags are mutual inverses: rtags[tags[i]] == i for every i.
    np.testing.assert_array_equal(rtags[tags], np.arange(N))


def test_restart_restores_per_particle_data_in_tag_order(tmp_path):
    """A one-time GSD restart restores per-particle data by tag, in current order."""
    sim_prev = _simulation(_unsorted_positions(), sort=True)
    sim_prev.run(2)

    saved = {}

    def grab(_positions, vel_mass, rtags, *_):
        saved["vx"] = _read(vel_mass)[:, 0]
        saved["rtags"] = _read(rtags).astype(int)

    _forward(SystemView(sim_prev._cpp_sys), AccessMode.Read, grab)

    gsd_path = str(tmp_path / "restart.gsd")
    hoomd.write.GSD.write(state=sim_prev.state, filename=gsd_path, mode="wb")

    # --- restart: fresh state from GSD ---
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_gsd(filename=gsd_path)
    view = SystemView(sim._cpp_sys)

    current_tags = _tags(view)
    perm = saved["rtags"][current_tags]

    # A naive copy would land data in the wrong slots.
    assert not np.allclose(
        saved["vx"], current_tags.astype(float)
    ), "restart state already matches the saved order"

    # Tag-ordered restore, written once in the current order.
    def restore(_positions, vel_mass, *_):
        _writable(vel_mass)[:, 0] = saved["vx"][perm]

    _forward(view, AccessMode.Overwrite, restore)

    # Each particle recovered its own (tag-encoded) value.
    np.testing.assert_allclose(_read_bundle(view)["vx"], current_tags.astype(float))
