#include <gtest/gtest.h>

#include <random>
#include <blust/blust.hpp>
using namespace blust;


TEST(Matrix, TestMultiplicationByConst)
{
    matrix<int> m({
        { 1, 2}, 
        {-1, 3}, 
        { 3, 1}
    });

    int k = 2;

    ASSERT_EQ(m * k, matrix<int>({
        { 2, 4},
        {-2, 6},
        { 6, 2}
    }));
}


TEST(Matrix, TestMultiplyMatrices)
{
    matrix<int> m({
        { 1, 2}, 
        {-1, 3}, 
        { 3, 1}
    });

    matrix<int> d({
        { 5, 3,  5, 2,  1},
        {-1, 5, 10, 8, -7}
    });

    ASSERT_EQ(m * d, matrix<int>({
        { 3, 13, 25, 18, -13},
        {-8, 12, 25, 22, -22},
        {14, 14, 25, 14, -4}
    }));
}

TEST(Matrix, TestMultiplyMatrixByVector)
{
    matrix<int> m({
        { 5, 3,  5, 2,  1},
        {-1, 5, 10, 8, -7}
    });

    std::vector<int> v({2, 7, -5, 9, 1});

    auto res = m * v;
    auto expect = std::vector<int>({25, 48});
    ASSERT_EQ(res, expect);
}

TEST(Matrix, SpeedTestVectorMultiplication)
{
    std::uniform_int_distribution<size_t> dist(2, 512);
    std::mt19937 rd(0x144258);

    for (size_t i = 0; i < 512; i++)
    {
        matrix<int> m({dist(rd), dist(rd)});
        std::vector<int> v(m.cols());

        const size_t size = m.size();
        for (size_t i = 0; i < size; i++)
            m(i) = dist(rd);
        
        for(size_t i = 0; i < v.size(); i++)
            v[i] = dist(rd);
        
        auto r = m * v;

        // ASSERT_TRUE(r.size() == m.rows());
    }
}