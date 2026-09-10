#include "pattern.hpp"

#include <algorithm>
#include <utility>

#include "options.hpp"

using json = nlohmann::json;

std::map<std::string, Pattern> patterns;

void from_json(const json &j, Pattern &p) {
    int totalmessages = 0;

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
        p.comm_partners_count = total;
        nneighbors_max = std::max(p.comm_partners.rbegin()->first, nneighbors_max);
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
        data_sent_max = std::max(p.buffer_size.rbegin()->first, data_sent_max);
    }

    // --- dist_to_neighbors ---
    for (auto &[outer_k, inner_obj] : j.at("dist_to_neighbors").items()) {
        int outer_key = std::stoi(outer_k);
        std::map<int, int> temp;

        for (auto &[inner_k, inner_v] : inner_obj.items()) {
            int amount = inner_v.get<int>();
            temp[std::stoi(inner_k)] = amount;
            totalmessages += amount / 2;
        }

        double total = 0.0;
        for (auto &[k, v] : temp) total += v;

        std::map<int, double> norm_inner;
        for (auto &[k, v] : temp) {
            norm_inner[k] = (total > 0) ? (double)v / total : 0.0;
        }

        p.dist_to_neighbors[outer_key] = std::move(norm_inner);
    }

    p.pattern_count = j.value("pattern_count", 1);
    p.message_count = j.value("message_count", totalmessages);
}
