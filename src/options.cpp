#include "options.hpp"

int nsamples = 25;
int niterations = 1;

std::string filepath = "";
distribution_t distribution_type = EMPIRICAL;
halo_t halo_type = EXPORT;
comm_t comm_type = MPIADVANCE;
std::string crs = "DEFAULT";
bool barrier = false;
int seed = -1;
bool unique_seed = false;
int data_sent_max = -1;
int nneighbors_max = -1;

int nosy_percent = 0;
int nosy_time_ms = 0;
