# Open the HLS project
open_project vitis_hls_project

# Set the top function for synthesis
set_top ViT_compute

# Add source and testbench files
add_files src/add.cpp
add_files src/attention.cpp
add_files src/conv.cpp
add_files src/gelu.cpp
add_files src/layernorm.cpp
add_files src/linear.cpp
add_files src/moe.cpp
add_files src/ViT_compute.cpp
add_files -tb testbench/e2e.cpp -cflags "-Wno-unknown-pragmas -Wno-psabi" -csimflags "-Wno-unknown-pragmas -Wno-psabi"

# Open the solution for HLS
open_solution "solution1" -flow_target vitis

# Set the FPGA part (for example, ZCU102)
set_part {xczu9eg-ffvb1156-2-e}

# Create a clock constraint (if needed)
create_clock -period 300MHz -name default

# Run only C simulation
# csim_design

# Start co-simulation
cosim_design -tool auto -trace_level all

# Run HLS synthesis (if not already done)
csynth_design
