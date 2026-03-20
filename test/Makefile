SIM ?= verilator
TOPLEVEL_LANG ?= verilog
SIM_BUILD ?= sim_build
COMPILE_ARGS += -I$(PWD)/../src --timing -Wno-fatal

VERILOG_SOURCES += $(wildcard $(PWD)/../src/*.v)

TOPLEVEL = top_level_module
MODULE = test

include $(shell cocotb-config --makefiles)/Makefile.sim
