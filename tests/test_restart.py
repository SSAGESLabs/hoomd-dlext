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

import numpy as np
import pytest

hoomd = pytest.importorskip("hoomd")
dlext = pytest.importorskip("hoomd.dlext")

from hoomd.dlext import (  # noqa: E402
    AccessMode,
    DLExtSampler,
    SystemView,
)

from helpers import (  # noqa: E402
    LOC,
    N,
    generate_simulation,
    read_tensor,
    writable_tensor,
)


def _tags(view):
    with view:
        return read_tensor(dlext.tags(view, LOC, AccessMode.Read)).astype(int)


def _forward(view, mode, callback):
    DLExtSampler(view, lambda *args: None, LOC, AccessMode.Read).forward_data(
        callback, LOC, mode, 0
    )


def _read_bundle(view):
    out = {}

    def callback(_positions, vel_mass, rtags, *_):
        out["vx"] = read_tensor(vel_mass)[:, 0]
        out["rtags"] = read_tensor(rtags).astype(int)

    _forward(view, AccessMode.Read, callback)
    return out


def test_tags_accessor_inverts_rtags_after_reorder():
    """``dlext.tags`` is reorder-aware and the inverse of ``rtags``."""
    sim = generate_simulation(sort=True)
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
    sim_prev = generate_simulation(sort=True)
    sim_prev.run(2)

    saved = {}

    def grab(_positions, vel_mass, rtags, *_):
        saved["vx"] = read_tensor(vel_mass)[:, 0]
        saved["rtags"] = read_tensor(rtags).astype(int)

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
        writable_tensor(vel_mass)[:, 0] = saved["vx"][perm]

    _forward(view, AccessMode.Overwrite, restore)

    # Each particle recovered its own (tag-encoded) value.
    np.testing.assert_allclose(_read_bundle(view)["vx"], current_tags.astype(float))
