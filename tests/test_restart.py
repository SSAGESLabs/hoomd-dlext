# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md
"""
Restart regression tests for hoomd-dlext.

HOOMD reorders particles internally so that a particle's local slot index changes
between runs while its global tag does not (spatial sort, done by default on GPU;
forced here with a ``ParticleSorter`` so it is reproducible on CPU). Restoring saved
per-particle data into the right slots therefore needs the slot->tag map.

hoomd-dlext exposes that map two ways: ``rtags`` (tag->slot) rides in the per-step
callback bundle, and ``tags`` (slot->tag) is available via the standalone ``dlext.tags``
accessor inside a ``with sysview:`` block. The slot->tag map is needed only at restore,
which happens once at the start of a run -- the per-step path keeps writing the bias in
slot order, unpermuted -- so it lives in the standalone accessor rather than burdening
every step.

These tests pin that the accessor is a correct, reorder-aware inverse of ``rtags``, and
that a one-time, tag-ordered restore through the plugin's Overwrite path lands data in
the slot order HOOMD is using at restore time -- where a naive slot-by-slot copy does
not. The matching red->green restart test lives in PySAGES, whose ``restore`` performs
the slot-by-slot copy these tests show to be wrong.
"""

import ctypes

import numpy as np
import pytest

hoomd = pytest.importorskip("hoomd")
dlext = pytest.importorskip("hoomd.dlext")

from hoomd.dlext import AccessLocation, AccessMode, DLExtSampler, SystemView  # noqa: E402

N = 8
LOC = AccessLocation.OnHost


class _DLPackCapsule:
    """Adapt a raw DLPack ``dltensor`` capsule for numpy. Host tensors are CPU device 0."""

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
    """A writable view over the capsule's memory, for the one-time Overwrite restore."""
    array = np.from_dlpack(_DLPackCapsule(capsule))
    address = array.__array_interface__["data"][0]
    pointer = ctypes.cast(address, ctypes.POINTER(np.ctypeslib.as_ctypes_type(array.dtype)))
    return np.ctypeslib.as_array(pointer, shape=array.shape)


def _nve_method():
    """An NVE integration method, across HOOMD versions (renamed in v4)."""
    methods = hoomd.md.methods
    cls = getattr(methods, "ConstantVolume", None) or methods.NVE
    return cls(filter=hoomd.filter.All())


def _reversed_layout():
    """Lay tags out as the reverse of spatial (x) order, so a spatial sort permutes them."""
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


def _standalone_tags(view):
    """slot->tag via the standalone accessor (the map needed to restore by identity)."""
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
    """``dlext.tags`` (slot->tag) is reorder-aware and the inverse of ``rtags``."""
    sim = _simulation(_reversed_layout(), sort=True)
    sim.run(2)
    view = SystemView(sim._cpp_sys)

    tags = _standalone_tags(view)
    assert not np.array_equal(tags, np.arange(N)), (
        "scenario did not reorder particles; the test would be vacuous"
    )

    rtags = _read_bundle(view)["rtags"]
    # tags and rtags are mutual inverses: rtags[tags[i]] == i for every slot i.
    np.testing.assert_array_equal(rtags[tags], np.arange(N))


def test_restart_restores_per_particle_data_in_tag_order(tmp_path):
    """A one-time GSD restart restores per-particle data by tag, in current slot order."""
    # --- save leg: a reordered (sorted) system ---
    sim_prev = _simulation(_reversed_layout(), sort=True)
    sim_prev.run(2)

    saved = {}

    def grab(_positions, vel_mass, rtags, *_):
        saved["payload"] = _read(vel_mass)[:, 0]  # per-particle data, save-time slot order
        saved["rtags"] = _read(rtags).astype(int)  # tag -> slot, at save time

    _forward(SystemView(sim_prev._cpp_sys), AccessMode.Read, grab)

    gsd_path = str(tmp_path / "restart.gsd")
    hoomd.write.GSD.write(state=sim_prev.state, filename=gsd_path, mode="wb")

    # --- restart leg: fresh state from GSD ---
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_gsd(filename=gsd_path)
    view = SystemView(sim._cpp_sys)

    current_tags = _standalone_tags(view)  # slot -> tag, read once at restart time
    perm = saved["rtags"][current_tags]  # current slot -> matching save-time slot

    # A naive slot-by-slot copy would land data in the wrong slots.
    assert not np.allclose(saved["payload"], current_tags.astype(float)), (
        "scenario did not exercise a slot remap; the test would be vacuous"
    )

    # Tag-ordered restore, written once in the current slot order via the Overwrite path.
    def restore(_positions, vel_mass, *_):
        _writable(vel_mass)[:, 0] = saved["payload"][perm]

    _forward(view, AccessMode.Overwrite, restore)

    # Each particle recovered its own (tag-encoded) value.
    np.testing.assert_allclose(_read_bundle(view)["vx"], current_tags.astype(float))

    # The data<->tag association survives a subsequent HOOMD reorder.
    sim.operations.tuners.append(
        hoomd.tune.ParticleSorter(trigger=hoomd.trigger.Periodic(1))
    )
    integrator = hoomd.md.Integrator(dt=0.0)
    integrator.methods.append(_nve_method())
    sim.operations.integrator = integrator
    sim.run(2)
    np.testing.assert_allclose(
        _read_bundle(view)["vx"], _standalone_tags(view).astype(float)
    )
