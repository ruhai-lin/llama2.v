#!/usr/bin/env python3
import math
import os
import struct


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIN_PATH = os.path.join(ROOT, "src", "LUTs", "rope_sin_lut.hex")
COS_PATH = os.path.join(ROOT, "src", "LUTs", "rope_cos_lut.hex")
MAX_SEQ_LEN = 512
FREQS = [1.0, 0.1, 0.01, 0.001]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def main():
    os.makedirs(os.path.dirname(SIN_PATH), exist_ok=True)
    with open(SIN_PATH, "w", encoding="ascii") as sin_f, open(COS_PATH, "w", encoding="ascii") as cos_f:
        for freq in FREQS:
            for pos in range(MAX_SEQ_LEN):
                angle = pos * freq
                sin_f.write(f"{f32_bits(math.sin(angle)):08x}\n")
                cos_f.write(f"{f32_bits(math.cos(angle)):08x}\n")


if __name__ == "__main__":
    main()
