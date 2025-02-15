#pragma once

#include "namespaces.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <iostream>


namespace std
{
    // Vector printing
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
    {
        auto last = v.end() - 1;
        os << '[';
        for (auto& e : v)
        {
            // compare the addresses 
            if (&e == &*last)
                os << e;
            else
                os << e << ", ";
        }
        os << ']';

        return os;
    }
}

START_BLUST_NAMEPSPACE
namespace utils
{
    
    // Source: https://stackoverflow.com/posts/59522794/revisions
    // Author: HolyBlackCat
    namespace type_names
    {
        template <typename T>
        constexpr const auto &RawTypeName()
        {
            #ifdef _MSC_VER
            return __FUNCSIG__;
            #else
            return __PRETTY_FUNCTION__;
            #endif
        }
    
        struct RawTypeNameFormat
        {
            std::size_t leading_junk = 0, trailing_junk = 0;
        };
    
        // Returns `false` on failure.
        inline constexpr bool GetRawTypeNameFormat(RawTypeNameFormat *format)
        {
            const auto &str = RawTypeName<int>();
            for (std::size_t i = 0;; i++)
            {
                if (str[i] == 'i' && str[i+1] == 'n' && str[i+2] == 't')
                {
                    if (format)
                    {
                        format->leading_junk = i;
                        format->trailing_junk = sizeof(str)-i-3-1; // `3` is the length of "int", `1` is the space for the null terminator.
                    }
                    return true;
                }
            }
            return false;
        }
    
        inline static constexpr RawTypeNameFormat format =
        []{
            static_assert(GetRawTypeNameFormat(nullptr), "Unable to figure out how to generate type names on this compiler.");
            RawTypeNameFormat format;
            GetRawTypeNameFormat(&format);
            return format;
        }();

        // Returns the type name in a `std::array<char, N>` (null-terminated).
        template <typename T>
        constexpr auto CexprTypeName()
        {
            constexpr std::size_t len = 
                sizeof(type_names::RawTypeName<T>()) 
                - type_names::format.leading_junk 
                - type_names::format.trailing_junk;
            
            std::array<char, len> name{};
            for (std::size_t i = 0; i < len-1; i++)
                name[i] = type_names::RawTypeName<T>()[i + type_names::format.leading_junk];
            return name;
        }
    }
    

    /**
     * @brief Returns a human readable c-string name of the `T` 
     */
    template <typename T>
    const char *TypeName()
    {
        static constexpr auto name = type_names::CexprTypeName<T>();
        return name.data();
    }
    
}
END_BLUST_NAMESPACE