#!/usr/bin/env python3
import math
import os
import struct


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PATH = os.path.join(ROOT, "src", "LUTs", "rsqrt_lut.hex")


def f32_bits(value):
    if not math.isfinite(value):
        value = 3.4028235e38
    if value > 3.4028235e38:
        value = 3.4028235e38
    return struct.unpack("<I", struct.pack("<f", value))[0]


def lut_value(index):
    exp = (index >> 4) & 0xFF
    mant = index & 0xF
    probe = struct.unpack("<f", struct.pack("<I", (exp << 23) | (mant << 19)))[0]
    if not math.isfinite(probe) or probe <= 0.0:
        return 0.0
    return min(1.0 / math.sqrt(probe), 3.4028235e38)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="ascii") as f:
        for idx in range(4096):
            f.write(f"{f32_bits(lut_value(idx)):08x}\n")


if __name__ == "__main__":
    main()
