# Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-level Sparsity via Mixture-of-Experts

Rishov Sarkar<sup>1</sup>, Hanxue Liang<sup>2</sup>, Zhiwen Fan<sup>2</sup>, Zhangyang Wang<sup>2</sup>, Cong Hao<sup>1</sup>

<sup>1</sup>School of Electrical and Computer Engineering, Georgia Institute of Technology  
<sup>2</sup>School of Electrical and Computer Engineering, University of Texas at Austin

ICCAD 2023 [paper](https://arxiv.org/abs/2305.18691)

## Overview

![Edge-MoE overall architecture](images/edge-moe-arch.svg)

This is **Edge-MoE**, the *first end-to-end* FPGA accelerator for *multi-task ViT* with a rich collection of architectural innovations.

## Usage

Run the Vitis HLS with the ``vitis_hls.tcl`` script.
```bash
vitis_hls -f vitis_hls.tcl
```

Run the Vivado
```bash
vivado -mode batch -source vivado.tcl
```

## Note

### Updated vitis_hls script with organized structure

```tcl
# Open your HLS project
open_project vitis_hls_project

# Set the top function for synthesis
set_top ViT_compute

# Add all source and testbench files
add_files src/add.cpp
add_files src/attention.cpp
add_files src/conv.cpp
add_files src/gelu.cpp
add_files src/layernorm.cpp
add_files src/linear.cpp
add_files src/moe.cpp
add_files src/ViT_compute.cpp
add_files -tb testbench/e2e.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"

# Open the solution and set the flow target to vitis
open_solution "solution1" -flow_target vitis

# Set the FPGA part (for example, ZCU102)
set_part {xczu9eg-ffvb1156-2-e}

# Create a clock constraint (if needed)
create_clock -period 300MHz -name default

# Run synthesis (if not already done)
csynth_design

# Set up co-simulation with simulation flags
cosim_design -tool xsim -trace_level none

```

### Issues with <gmp.h>

#### Fixing the use of gmp.h for co-simulation

The issue of the [Vitis HLS 2021.x - Use of gmp.h for Co-simulation](https://adaptivesupport.amd.com/s/article/Use-of-gmp-h-for-Co-simulation?language=en_US)
The error ``/include/file.h:244:2: error: ‘__gmp_const’ does not name a type`` occurs because of ``#include <gmp.h>``. Thus, while creating the symlink to the gmp header file, we also have to fix the following lines before including any Vitis HLS files to remove the error for all of the involved files while running co-simulation.

```bash
#include <gmp.h> 
#define __gmp_const const
```
For example, the header file ``/tools/Xilinx/Vitis_HLS/2022.1/include/mpfr.h`` also contains the line ``#include <gmp.h>``, so we have to add the ``#define __gmp_const const`` followed by that.


#### Fixing the path to the <gmp.h>

If the library file `libgmp.so` is present but the header file `gmp.h` is missing, the `libgmp-dev` package might not have been installed correctly. Follow these steps to resolve this:

---

##### 1. **Reinstall the GMP Development Package**
Run the following command to ensure that the development package is properly installed:
```bash
sudo apt update
sudo apt install --reinstall libgmp-dev
```

This package includes both the library and the header files required for GMP.

---

##### 2. **Verify the Header File**
Check again for the presence of `gmp.h`:
```bash
ls /usr/include/gmp.h
```

---

##### 3. **Alternative Locations**
If `gmp.h` is still not found, it might be installed in an unexpected location. Search for it:
```bash
sudo find / -name "gmp.h" 2>/dev/null
```
This command searches the entire filesystem for `gmp.h`.

---

##### 4. **Create a Symlink (if needed)**
If `gmp.h` is located in a non-standard directory (e.g., `/usr/local/include`), you can create a symlink to `/usr/include`:
```bash
sudo ln -s /path/to/actual/gmp.h /usr/include/gmp.h
```
Replace `/path/to/actual/gmp.h` with the actual path returned by the search.


##### 5. **Update Include Path in Vitis HLS**
If `gmp.h` is located in a directory other than `/usr/include`, ensure you add the correct include path in your Vitis HLS settings. For example, if `gmp.h` is in `/usr/local/include`, use:
```bash
-I/usr/local/include
```
