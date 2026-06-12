# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md

"""Tests for the Python DLPack protocol wrapper."""

import numpy as np
import pytest

pytest.importorskip("hoomd")
dlext = pytest.importorskip("hoomd.dlext")

from helpers import LOC, generate_simulation  # noqa: E402
from hoomd.dlext import AccessMode, DLPackTensor, SystemView  # noqa: E402


def test_property_getter_returns_dlpack_tensor():
    sim = generate_simulation()
    view = SystemView(sim._cpp_sys)

    with view:
        tensor = dlext.tags(view, LOC, AccessMode.Read)
        assert isinstance(tensor, DLPackTensor)
        assert tensor.__dlpack_device__() == (1, 0)
        np.from_dlpack(tensor)
        with pytest.raises(RuntimeError, match="already been consumed"):
            tensor.__dlpack__()


def test_property_tensor_is_invalidated_on_context_exit():
    sim = generate_simulation()
    view = SystemView(sim._cpp_sys)

    with view:
        tensor = dlext.tags(view, LOC, AccessMode.Read)

    with pytest.raises(RuntimeError, match="already been consumed"):
        tensor.__dlpack__()
