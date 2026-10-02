#include "options.hpp"

int nsamples = 25;
int nwindow = 10;
int ntimed = 1000;
double calc_budget_s = -1.0;

std::string filepath = "";
std::vector<std::string> selected_patterns;
warmup_t warmup_mode = WARMUP_NONE;
distribution_t distribution_type = EMPIRICAL;
halo_t halo_type = EXPORT;
comm_t comm_type = MPIADVANCE;
std::string crs = "DEFAULT";
bool barrier = false;
bool distinct_neighbors = false;
bool verify = false;
int seed = -1;
bool unique_seed = false;
int data_sent_max = -1;

int nosy_percent = 0;
int nosy_time_ms = 0;
