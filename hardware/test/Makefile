SIM ?= icarus
TOPLEVEL_LANG ?= verilog
SIM_BUILD ?= sim_build_local
COMPILE_ARGS += -I$(PWD)/../src

VERILOG_SOURCES += $(wildcard $(PWD)/../src/*.v)

TOPLEVEL = top_level_module
MODULE = test

include $(shell cocotb-config --makefiles)/Makefile.sim
