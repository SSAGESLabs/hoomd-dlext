// SPDX-License-Identifier: MIT
// This file is part of `hoomd-dlext`, see LICENSE.md

#pragma once


#if !defined(DEFAULT_VISIBILITY)
#if defined(WIN32) || defined(_WIN32)
#define DEFAULT_VISIBILITY __declspec(dllexport)
#else
#define DEFAULT_VISIBILITY __attribute__((visibility("default")))
#endif
#endif

#define INVOKE(object, member_ptr) ((object).*(member_ptr))


#include <vector>


namespace utils
{

template <typename T>
constexpr void maybe_unused(T&&) { }

//template<typename F, typename T>
//void flatten_iterate(F f, const std::vector< std::vector<T> >& outter) {
//    for (const auto& inner : outter) {
//        for (const auto& value : inner) {
//            f(value);
//        }
//    }
//}

}
