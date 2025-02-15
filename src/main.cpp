#include <blust/blust.hpp>
#include <iostream>

int main()
{
    using namespace blust;
    matrix<int> m({2, 3}, 1);
    matrix<double> md({3, 5}, 6);

    for (size_t i = 0; i < m.size(); i++)
        m(i / m.cols(), i % m.cols()) = i + 1;

    std::cout << m;
    std::cout << md;

    auto mr = m * md;

    std::cout << mr;
    std::cout << (md * 4.27);
    std::cout << m.T();

    md = m;
    std::cout << md;

    std::cout << mr[0] << '\n' << mr[1] << '\n';

    return 0;
}