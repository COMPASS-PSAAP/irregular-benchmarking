#include "sampling.hpp"

#include <random>
#include <vector>

#include "options.hpp"

namespace {

// Seeded from the `seed` option, which parseArgs() has already resolved (and
// rank-adjusted, if --unique-seed was given), so a run is reproducible.
std::mt19937 &global_rng() {
    static std::mt19937 rng(static_cast<unsigned int>(seed));
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
