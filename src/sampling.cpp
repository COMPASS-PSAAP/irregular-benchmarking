#include "sampling.hpp"

#include <chrono>
#include <random>
#include <vector>

namespace {

std::mt19937 &global_rng() {
    static std::mt19937 rng(static_cast<unsigned long>(
        std::chrono::system_clock::now().time_since_epoch().count()));
    return rng;
}

} // namespace

int sample_from_map(const std::map<int, double> &dist) {
    std::vector<int> keys;
    std::vector<double> weights;

    for (auto &[k, w] : dist) {
        keys.push_back(k);
        weights.push_back(w);
    }

    std::discrete_distribution<> d(weights.begin(), weights.end());
    return keys[d(global_rng())];
}
