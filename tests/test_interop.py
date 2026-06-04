# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md

"""
Test if hoomd's ``Simulation._cpp_sys`` can be converted back to
``std::shared_ptr<System>`` by ``SystemView`` to catch pybind11 ABI mismatches.
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
