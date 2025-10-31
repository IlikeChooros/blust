#include <gtest/gtest.h>
#include <blust/blust.hpp>

using namespace blust;

void test_mat_mul(number_t *a_data, number_t *b_data, number_t *c_data, size_t n, size_t m, size_t k)
{
    tensor r{{(int)n, (int)k}};
    auto r_data = r.data();

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < k; j++)
        {
            number_t sum = 0;
            for (size_t l = 0; l < m; l++)
            {
                sum += a_data[i * m + l] * b_data[l * k + j];
            }
            r_data[i * k + j] = sum;
        }
    }

    auto size = r.size();
    for (size_t i = 0; i < size; i++)
    {
        if (fabs(r_data[i] - c_data[i]) > 1e-2)
        {
            char buffer[128];
            snprintf(buffer, 128, "m=%lu, n=%lu, k=%lu (%lu,%lu) (test != result) %.5f != %.5f\n",
                     m, n, k, i / k, i % k, r_data[i], c_data[i]);
            throw std::runtime_error(buffer);
        }
    }
}

void fill_random(tensor &t)
{
    static std::mt19937 gen{std::random_device{}()};

    std::uniform_real_distribution<number_t> dist{0, 1};
    t.fill([&dist]()
           { return dist(gen); });
}

void test_vector_like(
    number_t *a_data, number_t *b_data, number_t *c_data,
    size_t n, std::function<number_t(number_t, number_t)> f)
{
    for (size_t i = 0; i < n; i++) {
        number_t r = f(a_data[i], b_data[i]);
        
        if (c_data[i] - r > 1e-2) {
            char buffer[128];
            snprintf(buffer, 128, "n=%lu (at %lu) (test != result) %.5f != %.5f\n",
                     n, i, r, c_data[i]);
            throw std::runtime_error(buffer);
        }
    }
}

TEST(OpsTest, MatMul)
{
    // Perform X multiplications with random sizes
    constexpr auto nMults = 100, minSize = 16, maxSize = 64;
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist{minSize, maxSize};
    cpu_ops ops;
    tensor t1, t2, r;
    int m, n, k;

    for (auto i = 0; i < nMults; i++)
    {
        m = dist(gen);
        n = dist(gen);
        k = dist(gen);

        t1 = tensor({m, n});
        t2 = tensor({n, k});
        fill_random(t1);
        fill_random(t2);

        r = ops.mat_mul(t1, t2);
        try
        {
            test_mat_mul(t1.data(), t2.data(), r.data(),
                         m, n, k);
        }
        catch (const std::runtime_error &e)
        {
            FAIL() << e.what();
        }
    }
}

TEST(OpsTest, SquarePowerOf2Matmul)
{
    constexpr auto max_pow = 6;
    cpu_ops ops;
    tensor t1, t2, r;

    for (int i = 0; i < max_pow; i++)
    {
        auto v = (1 << i);
        auto s = shape({v, v});
        t1 = tensor(s);
        t2 = tensor(s);
        fill_random(t1);
        fill_random(t2);
        r = ops.mat_mul(t1, t2);

        try
        {
            test_mat_mul(t1.data(), t2.data(), r.data(),
                         v, v, v);
        }
        catch (const std::runtime_error &e)
        {
            if (i < 6) {
                FAIL() << e.what() << t1 << t2 << r << '\n';
            } else {
                FAIL() << e.what() << "tensors aren't printable\n";
            }
        }
    }
}

enum VectorOp {ADD, SUB, MUL};
void testVectorLike(VectorOp op) {

    constexpr auto nOps = 100, minSize = 16, maxSize = 512;
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist{minSize, maxSize};
    cpu_ops ops(4); // allow multithreaded vector operations
    tensor t1, t2, r;
    int n, m;
    std::function<number_t(number_t, number_t)> f;
    std::function<tensor(tensor&, tensor&)> top;

    switch (op)
    {
    case SUB:
        f = [](number_t a, number_t b) { return a - b; };
        top = [&ops](tensor& t1, tensor& t2) { return ops.sub(t1, t2); };
        break;
    case MUL:
        f = [](number_t a, number_t b) { return a * b; };
        top = [&ops](tensor& t1, tensor& t2) { return ops.hadamard(t1, t2); };
        break;
    default:
        f = [](number_t a, number_t b) { return a + b; };
        top = [&ops](tensor& t1, tensor& t2) { return ops.add(t1, t2); };
        break;
    }

    for (auto i = 0; i < nOps; i++)
    {
        m = dist(gen);
        n = dist(gen);

        t1 = tensor({n, m});
        t2 = tensor({n, m});
        fill_random(t1);
        fill_random(t2);

        r = top(t1, t2);
        try
        {
            test_vector_like(t1.data(), t2.data(), r.data(),
                         m*n, f);
        }
        catch (const std::runtime_error &e)
        {
            if (n < 32 && m < 32) {
                FAIL() << e.what() << "t1:\n" << t1 << "t2:\n" << t2 << "r:\n" << r << '\n';
            } else {
                FAIL() << e.what() << "tensors aren't printable\n";
            }
        }
    }
}

TEST(OpsTest, TensorAdd) {
    testVectorLike(ADD);
}

TEST(OpsTest, TensorSub) {
    testVectorLike(SUB);
}

TEST(OpsTest, TensorHadamardMul) {
    testVectorLike(MUL);
}
