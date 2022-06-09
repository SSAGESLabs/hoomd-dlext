// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#ifndef CXX11_UTILS_H_
#define CXX11_UTILS_H_

#if !defined(DEFAULT_VISIBILITY)
#if defined(WIN32) || defined(_WIN32)
#define DEFAULT_VISIBILITY __declspec(dllexport)
#else
#define DEFAULT_VISIBILITY __attribute__((visibility("default")))
#endif
#endif

#define INVOKE(object, member_ptr) ((object).*(member_ptr))

namespace cxx11utils {

template <typename T> inline void maybe_unused(T &&) {}

} // namespace cxx11utils

#endif // CXX11_UTILS_H_
