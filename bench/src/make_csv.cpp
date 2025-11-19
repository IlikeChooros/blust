#include "defs.hpp"

using namespace blust;
typedef tensor_t(*matmul_func_t)(
    tensor_t& A, tensor_t& B,
    size_t m, size_t n, size_t k
);

static cpu_ops cOps;
static size_t m = 2048, k = 2048, n = 2048;
static std::vector<double> times(5, 0.0);

tensor_t base_line_mat_mul(
    tensor_t& A, tensor_t& B,
    size_t m, size_t n, size_t k) 
{
    tensor_t C({m, n}, 0.0);
    auto A_data = A.data();
    auto B_data = B.data();
    auto C_data = C.data();

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            number_t sum = 0;
            for (size_t p = 0; p < k; p++) {
                sum += A_data[i * k + p] * B_data[p * n + j];
            }
            C_data[i * n + j] = sum;
        }
    }

    return C;
}

tensor_t mat_mul(tensor& A, tensor& B, size_t mc, size_t kc, size_t nc) {
    return cOps.mat_mul(A, B, mc, kc, nc);
}

void benchmark_and_log(tensor_t& A, tensor_t& B, matmul_func_t f, size_t mc, size_t kc, size_t nc) {
    tensor_t r;

    // warm up
    r = f(A, B, mc, kc, nc);
    
    constexpr int n_trials = 5;
    for (int i = 0; i < n_trials; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        r = f(A, B, mc, kc, nc);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = (std::chrono::duration<double>(end - start).count());
    }
    
    // Compute statistics
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / n_trials;
    double variance = 0;
    for (auto t : times) variance += (t - mean) * (t - mean);
    double stddev = std::sqrt(variance / n_trials);
    double gflops = (2.0 * m * n * k) / (mean * 1e9);
    
    // CSV output
    std::cout << mc << "," << kc << "," << nc << ","
              << mean << "," << stddev << ","
              << gflops << "\n";
}

void make_csv(int argc, char** argv) {
    matmul_func_t func = mat_mul;
    int matmul_type_index = 1;
    int dim_start_index = 1;

    // See if the first argument is the matmul type
    if (argc > 1) {
        std::string first_arg = argv[1];
        if (first_arg == "baseline" || first_arg == "blust") {
            if (first_arg == "baseline") {
                func = base_line_mat_mul;
            } else {
                func = mat_mul;
            }
            matmul_type_index = 1;
            dim_start_index = 2;
        } else {
            matmul_type_index = 0;
            dim_start_index = 1;
        }

        if (argc >= 4 + matmul_type_index) {
            m = std::atoi(argv[dim_start_index]);
            k = std::atoi(argv[dim_start_index + 1]);
            n = std::atoi(argv[dim_start_index + 2]);

            // If failed
            if (m <= 0 || k <= 0 || n <= 0) {
                std::cerr << "Invalid matrix dimensions provided.\n";
                return;
            }
        }
    } 

    // CSV header
    std::cout << "MC,KC,NC,Time(s),StdDev(s),GFLOPS\n";
    tensor_t A({m, k});
    tensor_t B({k, n});
    utils::randomize(A.begin(), A.end(), A.size());
    utils::randomize(B.begin(), B.end(), B.size());

    for (auto mc : {64, 96, 128, 192, 256}) {
        for (auto kc : {96, 128, 192, 256, 384}) {
            for (auto nc : {512, 1024, 2048}) {
                benchmark_and_log(A, B, func, mc, kc, nc);
            }
        }
    }
}