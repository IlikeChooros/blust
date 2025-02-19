#pragma once

#include "namespaces.hpp"

#include <stdexcept>

START_BLUST_NAMESPACE

class InvalidMatrixSize : public std::runtime_error
{
    std::string _msg;
public:
    InvalidMatrixSize() : std::runtime_error("Invalid matrix size") {}

    /**
     * @brief Create expection object with matrix shape
     * @brief got shape of the matrix that we got (first = row, second = cols)
     * @brief expected shape
     */
    InvalidMatrixSize(std::pair<size_t, size_t> got, std::pair<size_t, size_t> expected) :
        std::runtime_error(
            "Got matrix: r=" + std::to_string(got.first) + " c=" + std::to_string(got.second) +
            ", expected: r=" + std::to_string(expected.first) + " c=" + std::to_string(expected.second)
        ) {}
};

END_BLUST_NAMESPACE