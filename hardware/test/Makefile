SIM ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(wildcard $(PWD)/../src/*.v)

TOPLEVEL = top_level_module
MODULE = test

include $(shell cocotb-config --makefiles)/Makefile.sim
