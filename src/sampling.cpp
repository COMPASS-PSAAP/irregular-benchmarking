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

int sample_interpolated(const std::vector<double> &edges, const std::vector<int> &counts) {
    if (edges.size() == 1) {
        return static_cast<int>(edges[0]);
    }
    std::discrete_distribution<std::size_t> pick(counts.begin(), counts.end());
    const std::size_t bin = pick(global_rng());
    const double upper = (bin + 1 < edges.size()) ? edges[bin + 1] : edges[bin] + (edges[1] - edges[0]);
    std::uniform_real_distribution<double> within(edges[bin], upper);
    return static_cast<int>(within(global_rng()));
}
