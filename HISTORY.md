# hoomd-dlext Release Notes

## v0.7.0

- Return Python DLPack protocol objects from property getters and sampler callbacks.
- Report the standard DLPack device ID `0` for CPU tensors, including in MPI simulations.

## v0.6.0

- Add support for HOOMD-blue 5, 6, and 7.
- Export `rtags` as read-only data from `CallbackHandler::forward_data`.
- Add `dlext.tags(...)` for tag lookup during restart + restore.
- Add restart regression tests covering tag-based particle data remapping.
- Expand CI coverage across HOOMD-blue 3.11.0, 4.9.1, 5.0.0, 6.1.1, and 7.0.1.
