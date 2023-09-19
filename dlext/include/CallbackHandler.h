// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef DLEXT_CALLBACKHANDLER_H_
#define DLEXT_CALLBACKHANDLER_H_

#include "DLExt.h"

namespace hoomd
{
namespace md
{
namespace dlext
{

#ifdef HOOMD2
using TimeStep = unsigned int;
#else
using TimeStep = uint64_t;
#endif

template <template <typename> class Wrapper>
class DEFAULT_VISIBILITY CallbackHandler {
public:
    //! Constructor
    CallbackHandler(SystemView& sysview)
        : _sysview { sysview }
    { }

    const SystemView& system_view() const { return _sysview; }

    //! Wraps the system positions, velocities, reverse tags, images and forces as
    //! DLPack tensors and passes them to the external function `callback`.
    //!
    //! The (non-typed) signature of `callback` is expected to be
    //!     callback(positions, velocities, rtags, images, forces, n)
    //! where `n` ìs an additional `TimeStep` parameter.
    //!
    //! The data for the particles information is requested at the given `location`
    //! and access `mode`. NOTE: Forces are always passed in readwrite mode.
    template <typename Callback>
    void forward_data(Callback callback, AccessLocation location, AccessMode mode, TimeStep n)
    {
        auto pos_capsule = Wrapper<PositionsTypes>::wrap(_sysview, location, mode);
        auto vel_capsule = Wrapper<VelocitiesMasses>::wrap(_sysview, location, mode);
        auto rtags_capsule = Wrapper<RTags>::wrap(_sysview, location, mode);
        auto img_capsule = Wrapper<Images>::wrap(_sysview, location, mode);
        auto force_capsule = Wrapper<NetForces>::wrap(_sysview, location, kReadWrite);

        callback(pos_capsule, vel_capsule, rtags_capsule, img_capsule, force_capsule, n);
    }

private:
    SystemView _sysview;
};

}  // namespace dlext
}  // namespace md
}  // namespace hoomd

#endif  // DLEXT_CALLBACKHANDLER_H_
