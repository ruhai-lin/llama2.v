# Llama-2 RTL Implementation Project

This project implements the core inference loop of Llama-2 (from [llama2.c](https://github.com/karpathy/llama2.c) and [Swan](https://github.com/turingmotors/swan)) in **Verilog-2001**, aiming for RTL-level performance and structure, ultimately targeting an end-to-end generation.

If you like this project, please consider starring ⭐ this repo as it is the easiest and best way to support it.

## Quick Start

### 1. Set up the Software Reference (Same as llama2.c)

Clone and build the software reference:

```bash
cd software
./run ../model/stories260K.bin -z ../model/tok512.bin
```

This will produce the expected output for verification.

### 2. Set up the Hardware (RTL) Simulation

1. Ensure you have **iveriverilog** and **cocotb** installed:

```bash
# Using pip
pip install cocotb
# For Icarus Verilog
sudo apt-get install iverilog libverilator-dev
# Or install via your preferred package manager
```

2. Run the hardware simulation:

```bash
cd hardware/test
make
```

This will execute `test.py` to simulate token generation on hardware.

You may also customize the prompt in `test.py` and run:

```bash
make MODULE=generate
```

to try your own prompt.

The simulation will take ~5 mins depending on local environment.
---

## Project Structure

- `software/` – Reference C implementation (llama2.c style)
- `model/` – Model weights and tokenizers (e.g., `stories260K.bin`, `tok512.bin`)
- `hardware/src/` – Verilog-2001 RTL design hierarchy
- `hardware/test/` – Simulation tests using cocotb and Icarus Verilog


## TODOs

This repository currently provides a hardware-oriented RTL prototype, not a finished synthesizable accelerator.
The current Verilog implementation uses behavioral code and cannot be synthesized into actual circuits yet.

1. **Replace behavioral MAC in `kernel_matmul.v`**
   - Implement real, synthesizable multiply-accumulate operations
   - Ensure proper width management and overflow handling

2. **Implement LUT-based `kernel_rmsnorm.v`**

3. **Implement LUT-based `kernel_softmax.v`**

4. **Create a new branch for Xilinx XC7A35T FPGA deployment**
   - Port the RTL design to Xilinx architecture
   - Optimize for FPGA constraints (resource usage, timing)
   - Add Xilinx-specific constraints and IP usage

5. **Create a new branch for Tiny Tapeout ASIC deployment**
   - Optimize for standard cell ASIC process
   - Meet Tiny Tapeout design rules and constraints
   - Focus on area efficiency and yield considerations
   - Submit to Tiny Tapeout program (if applicable)


## Contributions

Contributions to Swan are highly welcome. Please submit feedback and improvement suggestions through Issues and Pull Requests.