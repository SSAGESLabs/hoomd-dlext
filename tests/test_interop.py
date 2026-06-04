# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md
"""
Focused Python/C++ interop smoke tests.

Importing ``hoomd.dlext`` is not enough to catch pybind11 ABI mismatches: those
can fail only when HOOMD's Python ``_cpp_sys`` object is converted back to the
C++ ``std::shared_ptr<System>`` expected by ``SystemView``.
"""

import pytest

hoomd = pytest.importorskip("hoomd")
pytest.importorskip("hoomd.dlext")

from hoomd.dlext import SystemView  # noqa: E402


def test_system_view_accepts_hoomd_system():
    snap = hoomd.Snapshot()
    snap.particles.N = 3
    snap.particles.types = ["A"]
    snap.configuration.box = [10, 10, 10, 0, 0, 0]

    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(snap)

    view = SystemView(sim._cpp_sys)

    assert view.local_particle_number == 3
    assert view.global_particle_number == 3
