#!/usr/bin/env python3
import math
import os
import struct


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(ROOT, "src", "LUTs", "exp_lut.hex")


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def lut_value(index):
    exp = (index >> 4) & 0xFF
    mant = index & 0xF
    probe = struct.unpack("<f", struct.pack("<I", 0x80000000 | (exp << 23) | (mant << 19)))[0]
    if not math.isfinite(probe):
        return 0.0
    return math.exp(probe)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="ascii") as f:
        for idx in range(4096):
            f.write(f"{f32_bits(lut_value(idx)):08x}\n")


if __name__ == "__main__":
    main()
