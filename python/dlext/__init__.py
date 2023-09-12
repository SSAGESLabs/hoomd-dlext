# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md

# API exposed to Python
# isort: off

from ._api import (  # noqa: F401 # pylint: disable=E0401
    # Enums
    AccessLocation,
    AccessMode,
    # Classes
    SystemView,
    # Methods
    angular_momenta,
    charges,
    diameters,
    images,
    net_forces,
    net_torques,
    net_virial,
    orientations,
    positions_types,
    rtags,
    tags,
    velocities_masses,
    # Other attributes
    __version__,
)

# isort: on

kOnHost = AccessLocation.OnHost
if hasattr(AccessLocation, "OnDevice"):
    kOnDevice = AccessLocation.OnDevice

kRead = AccessMode.Read
kReadWrite = AccessMode.ReadWrite
kOverwrite = AccessMode.Overwrite


class DLExtSampler(_api.HalfStepHook):  # noqa: F821
    """
    This is a HalfStepHook derived class. It allows the user to set an `update_callback`
    method which will be passed to `forward_data`, along with a `location` and a `mode`.

    The (non-typed) signature  of `update_callback` is expected to be:

        update_callback(positions_types, velocities_masses, rtags, images, forces, n)

    where `n` is the current time step.
    """

    def __init__(self, sysview, update_callback, update_location, update_mode):
        from functools import partial

        super().__init__()
        callback_handle = _api.CallbackHandler(sysview)  # noqa: F821

        self.forward_data = callback_handle.forward_data
        self.update = partial(self.forward_data, update_callback, update_location, update_mode)
