# Open project and set top-level function
open_project vitis_hls_project
set_top ViT_compute

# Add source files
add_files src/add.cpp
add_files src/attention.cpp
add_files src/conv.cpp
add_files src/gelu.cpp
add_files src/layernorm.cpp
add_files src/linear.cpp
add_files src/moe.cpp
add_files src/ViT_compute.cpp -cflags "-I/tools/Xilinx/Vitis_HLS/2022.1/include"

# Add testbench file
add_files -tb testbench/e2e.cpp \
    -cflags "-I/tools/Xilinx/Vitis_HLS/2022.1/include -L/usr/lib/x86_64-linux-gnu" \
    -csimflags "-L/usr/lib/x86_64-linux-gnu -lpthread -lm"

# Open solution and set part
open_solution "solution1" -flow_target vitis

# Use the correct part for the Alveo U50 FPGA
set_part xcu50-fsvh2104-2-e
# set_part {xczu9eg-ffvb1156-2-e}

# Set platform for Alveo U50
# set_property platform "xilinx_u50_gen3x16_xdma_5_202210_1" [current_solution]

# Create clock with a 3.33 ns period (300 MHz)
create_clock -period 3.33 -name default

# Configure export settings for the generated IP
config_export -format ip_catalog -rtl verilog -version 1.0.0

# Configure interface for AXI alignment and latency
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512

# Step 1: C Simulation
puts "Starting C simulation..."
if {[catch {csim_design} result]} {
   puts "Error during C simulation: $result"
   exit 1
}
puts "C simulation completed successfully."

# Step 2: Synthesis
puts "Starting synthesis..."
if {[catch {csynth_design} result]} {
    puts "Error during synthesis: $result"
    exit 1
}
puts "Synthesis completed successfully."

# Step 3: RTL Co-simulation
puts "Starting co-simulation..."
if {[catch {cosim_design} result]} {
    puts "Error during co-simulation: $result"
    exit 1
}

# Export design
puts "Exporting design..."
if {[catch {export_design -format ip_catalog} result]} {
    puts "Error during export: $result"
    exit 1
}
puts "Co-simulation completed successfully."

