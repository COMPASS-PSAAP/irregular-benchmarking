set(SOURCES
    main.cpp
    cli.cpp
    options.cpp
    pattern.cpp
    matrix.cpp
    fine.cpp
    sampling.cpp
    benchmark.cpp
    )

set(HEADERS
    cli.hpp
    options.hpp
    pattern.hpp
    matrix.hpp
    fine.hpp
    sampling.hpp
    benchmark.hpp
    )

add_executable( irregular-benchmarking ${SOURCES} ${HEADERS} )

# zlib compresses the per-iteration timing files (--iter-prefix).
find_package(ZLIB REQUIRED)

target_link_libraries( irregular-benchmarking
        Cabana::Core
        Cabana::Grid
        MPI::MPI_CXX
        ZLIB::ZLIB
        )

# The commits the binary is built from, compiled in as one string that strings(1) finds
# (scripts/provenance.sh) and printed under -r. The dane build scripts pass both.
# Otherwise SOURCE_COMMIT is read from git, at configure time only, and only when the
# source is the top of its own checkout: a git-archive export inside another repository
# (a home directory that is a repo) must not report that repository's commit.
set(SOURCE_COMMIT "" CACHE STRING "Commit this source is; empty: read from git at configure time")
set(LOCALITY_AWARE_COMMIT "" CACHE STRING "locality_aware commit the binary links; empty: unknown")
set(resolved_commit "${SOURCE_COMMIT}")
if (resolved_commit STREQUAL "")
    set(resolved_commit "unknown (not a git checkout)")
    find_package(Git QUIET)
    if (GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} RESULT_VARIABLE git_status
                OUTPUT_VARIABLE git_top OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        get_filename_component(git_top "${git_top}" REALPATH)
        get_filename_component(source_dir "${PROJECT_SOURCE_DIR}" REALPATH)
        if (git_status EQUAL 0 AND git_top STREQUAL source_dir)
            execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    OUTPUT_VARIABLE resolved_commit OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE git_changes)
            if (NOT git_changes STREQUAL "")
                string(APPEND resolved_commit "-dirty")
            endif()
        endif()
    endif()
endif()
set(resolved_la_commit "${LOCALITY_AWARE_COMMIT}")
if (resolved_la_commit STREQUAL "")
    set(resolved_la_commit "unknown")
endif()
message(STATUS "Provenance: source ${resolved_commit}, locality_aware ${resolved_la_commit}")
target_compile_definitions(irregular-benchmarking PRIVATE
        "BENCH_SOURCE_COMMIT=\"${resolved_commit}\""
        "LOCALITY_AWARE_COMMIT=\"${resolved_la_commit}\"")

#
##CALIPER_start
#find_path(CALIPER_INCLUDE_DIR caliper/cali.h
#        PATHS
#        ${SPACK_PREFIX}/include
#)
#
#find_library(CALIPER_LIBRARY caliper
#        PATHS
#        ${SPACK_PREFIX}/lib
#)
#
## Check if we found Caliper
#if (CALIPER_INCLUDE_DIR AND CALIPER_LIBRARY)
#    message(STATUS "Found Caliper: ${CALIPER_LIBRARY}")
#    set(CALIPER_FOUND TRUE)
#else ()
#    message(FATAL_ERROR "Caliper not found!")
#endif ()
#
#set(CALIPER_INCLUDE_DIRS ${CALIPER_INCLUDE_DIR})
#set(CALIPER_LIBRARIES ${CALIPER_LIBRARY})
#
#include_directories(${CALIPER_INCLUDE_DIR})
##CALIPER_end
#

#TCLAP_start
find_path(TCLAP_INCLUDE_DIR tclap/CmdLine.h
        PATHS
        ${SPACK_PREFIX}/include
)

# Check if we found TCLAP
if (TCLAP_INCLUDE_DIR)
    message(STATUS "Found TCLAP: ${TCLAP_INCLUDE_DIR}")
    set(TCLAP_FOUND TRUE)
else ()
    message(FATAL_ERROR "TCLAP not found!")
endif ()
set(TCLAP_INCLUDE_DIRS ${TCLAP_INCLUDE_DIR})

include_directories(${TCLAP_INCLUDE_DIR})
#TCLAP_end

# nlohmann_json_start
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS
        ${SPACK_PREFIX}/include
)

# Check if we found nlohmann/json
if (NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "Found nlohmann/json: ${NLOHMANN_JSON_INCLUDE_DIR}")
    set(NLOHMANN_JSON_FOUND TRUE)
else ()
    message(FATAL_ERROR "nlohmann/json not found!")
endif ()
set(NLOHMANN_JSON_INCLUDE_DIRS ${NLOHMANN_JSON_INCLUDE_DIR})

include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
# nlohmann_json_end

#target_link_libraries( irregular-benchmarking PRIVATE ${CALIPER_LIBRARY} )

install(TARGETS irregular-benchmarking DESTINATION ${CMAKE_INSTALL_BINDIR})
