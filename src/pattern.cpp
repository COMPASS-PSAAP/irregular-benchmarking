#include "pattern.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "options.hpp"

using json = nlohmann::json;

std::map<std::string, Pattern> patterns;

void from_json(const json &j, Pattern &p) {
    // --- comm_partners ---
    {
        std::map<int, int> temp;
        for (auto &[k, v] : j.at("comm_partners").items()) {
            temp[std::stoi(k)] = v.get<int>();
        }

        // fill-forward
        int last_nonzero = 0;
        for (auto &[key, value] : temp) {
            if (value != 0) {
                last_nonzero = value;
            } else if (last_nonzero != 0) {
                value = last_nonzero;
            }
        }

        // normalize
        double total = 0.0;
        for (auto &[k, v] : temp) total += v;
        for (auto &[k, v] : temp) {
            p.comm_partners[k] = (total > 0) ? (double)v / total : 0.0;
        }
    }

    // --- buffer_size ---
    {
        std::map<int, int> temp;
        for (auto &[k, v] : j.at("buffer_size").items()) {
            temp[std::stoi(k)] = v.get<int>();
        }

        // fill-forward
        int last_nonzero = 0;
        for (auto &[key, value] : temp) {
            if (value != 0) {
                last_nonzero = value;
            } else if (last_nonzero != 0) {
                value = last_nonzero;
            }
        }

        // normalize
        double total = 0.0;
        for (auto &[k, v] : temp) total += v;
        for (auto &[k, v] : temp) {
            p.buffer_size[k] = (total > 0) ? (double)v / total : 0.0;
        }

        int largest = p.buffer_size.rbegin()->first;
        if (j.contains("buffer_size_bins")) {
            const std::string bins = j.at("buffer_size_bins").get<std::string>();
            if (bins != "lower_edge") {
                throw std::runtime_error("buffer_size_bins must be \"lower_edge\", not \"" + bins + "\"");
            }
            p.spread_sizes = true;
            for (auto &[k, v] : temp) {
                p.size_edges.push_back(k);
                p.size_counts.push_back(v);
            }
            // The last bin is taken to be as wide as the first (sample_interpolated).
            if (p.size_edges.size() > 1) {
                largest += static_cast<int>(p.size_edges[1] - p.size_edges[0]);
            }
        }
        data_sent_max = std::max(largest, data_sent_max);
    }

    // --- dist_to_neighbors ---
    for (auto &[outer_k, inner_obj] : j.at("dist_to_neighbors").items()) {
        int outer_key = std::stoi(outer_k);
        std::map<int, int> temp;

        for (auto &[inner_k, inner_v] : inner_obj.items()) {
            temp[std::stoi(inner_k)] = inner_v.get<int>();
        }

        double total = 0.0;
        for (auto &[k, v] : temp) total += v;

        std::map<int, double> norm_inner;
        for (auto &[k, v] : temp) {
            norm_inner[k] = (total > 0) ? (double)v / total : 0.0;
        }

        p.dist_to_neighbors[outer_key] = std::move(norm_inner);
    }
}
