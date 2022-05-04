#include "cholesky_decomposition.h"

#include <algorithm>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <chrono>
#include <benchmark/benchmark.h>


auto generate_matrix(int size = 10) -> std::vector<double> {
    std::vector<double> result(size * size);
    srand(100);

    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            result[i * size + j] = std::rand();
            result[j * size + i] = result[i * size + j];
        }
    }

    for (int i = 0; i < size; i++) {
        double sum = 0;
        result[i * size + i] = 0;
        for (int j = 0; j < size; j++) {
            sum += result[i * size + j];
        }
        result[i * size + i] = sum;
    }

    return result;
}


static void BM_Choletsky_decomposition_base(benchmark::State& state) {
    int N = state.range(0);
    auto A = generate_matrix(N);
    std::vector<double> L(N * N, 0);
    for (auto _ : state) {
        Cholesky_Decomposition_Base(A.data(), L.data(), N);
    }
}

static void BM_Choletsky_decomposition_base_parallel(benchmark::State& state) {
    int N = state.range(0);
    auto A = generate_matrix(N);
    std::vector<double> L(N * N, 0);
    for (auto _ : state) {
        Cholesky_Decomposition_Base_Parallel(A.data(), L.data(), N);
    }
}

static void BM_Choletsky_decomposition_blocks(benchmark::State& state) {
    int N = state.range(0);
    auto A = generate_matrix(N);
    std::vector<double> L(N * N, 0);
    for (auto _ : state) {
        Cholesky_Decomposition(A.data(), L.data(), N);
    }
}


BENCHMARK(BM_Choletsky_decomposition_base)->Arg(500)->Arg(1000)->Arg(3000)->Arg(5000)->Arg(7500);
BENCHMARK(BM_Choletsky_decomposition_base_parallel)->Arg(500)->Arg(1000)->Arg(3000)->Arg(5000)->Arg(7500);
BENCHMARK(BM_Choletsky_decomposition_blocks)->Arg(500)->Arg(1000)->Arg(3000)->Arg(5000)->Arg(7500);

auto sanity_check() -> bool {
    bool success = true;

    const int N = 1000;
    auto A = generate_matrix(N);
    auto A1 = A;
    auto A2 = A;
    auto A3 = A;

    std::vector<double> L_ref(N * N, 0);
    auto L1 = L_ref;
    auto L2 = L_ref;
    auto L3 = L_ref;

    auto is_same = [&](std::vector<double>& vec1, std::vector<double>& vec2) { 
        for (int i = 0; i < vec1.size(); i++)
            if (vec1[i] - vec2[i] > 0.001) {
                return false;
            }

        return true;
    };

    Cholesky_Decomposition_Base(A.data(), L_ref.data(), N);
    Cholesky_Decomposition_Base_Parallel(A1.data(), L1.data(), N);
    if (!is_same(L_ref, L1)) {
        std::cout << "Cholesky_Decomposition_Base_Parallel function is not working right!"
                  << "\n";
        success = false;
    }

    Cholesky_Decomposition(A2.data(), L2.data(), N);
    if (!is_same(L_ref, L2)) {
        std::cout << "Cholesky_Decomposition_By_Blocks function is not working right!"
                  << "\n";
        success = false;
    }

    return success;
}

int main(int argc, char** argv) {
    std::cout << "========="
              << "\n";
    if (sanity_check()) {
        std::cout << "Algorithmic tests passed!"
                  << "\n";
    } else {
        std::cout << "Algorithmic tests failed!"
                  << "\n";
        return 1;
    }
    std::cout << "========="
              << "\n";
#ifdef _OPENMP
    printf_s("Compiled by an OpenMP-compliant implementation.\n");
#else 
    printf_s("OpenMP is not found.\n");
#endif

    

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
