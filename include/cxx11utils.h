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

#include <memory>

namespace cxx11utils
{

template <typename T>
using SPtr = std::shared_ptr<T>;

template <typename T>
inline void maybe_unused(T&&)
{ }

// `make_unique` for C++11
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace cxx11utils

#endif  // CXX11_UTILS_H_
