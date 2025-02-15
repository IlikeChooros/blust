#pragma once

#include "namespaces.hpp"

#include <stdexcept>

START_BLUST_NAMEPSPACE

class InvalidMatrixSize : public std::exception
{
    std::string _msg;
public:
    InvalidMatrixSize() : _msg("Invalid matrix size") {}

    /**
     * @brief Create expection object with matrix shape
     * @brief got shape of the matrix that we got (first = row, second = cols)
     * @brief expected shape
     */
    InvalidMatrixSize(std::pair<size_t, size_t> got, std::pair<size_t, size_t> expected)
    {
        _msg  = "Got matrix: r=" + std::to_string(got.first) + " c=" + std::to_string(got.second);
        _msg += ", expected: r=" + std::to_string(expected.first) + " c=" + std::to_string(expected.second);
    }

    const char* what() {return _msg.c_str(); }
};

END_BLUST_NAMESPACE