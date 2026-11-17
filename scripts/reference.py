#!/usr/bin/env python3
"""Independent pure-Python Jacobi reference for small grids (hand check, validation test 1).

Python floats are IEEE doubles, and the update below uses the same summation order as the C++
code, ((N + S) + W) + E then * 0.25, so V0 must match this output bit for bit.

usage: reference.py --nx N --ny N --iters K --out FILE [--t-hot X] [--t-cold X] [--print]
"""
import argparse
import struct


def solve(nx, ny, iters, t_hot, t_cold):
    cur = [[t_hot if i == 0 else t_cold for _ in range(nx)] for i in range(ny)]
    nxt = [row[:] for row in cur]
    for _ in range(iters):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                nxt[i][j] = 0.25 * (((cur[i - 1][j] + cur[i + 1][j]) + cur[i][j - 1]) + cur[i][j + 1])
        cur, nxt = nxt, cur
    return cur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, required=True)
    p.add_argument("--ny", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t-hot", type=float, default=100.0)
    p.add_argument("--t-cold", type=float, default=0.0)
    p.add_argument("--print", action="store_true", help="print the grid (for the report)")
    a = p.parse_args()

    grid = solve(a.nx, a.ny, a.iters, a.t_hot, a.t_cold)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<qq", a.nx, a.ny))
        for row in grid:
            f.write(struct.pack(f"<{a.nx}d", *row))
    if a.print:
        print(f"{a.nx}x{a.ny} after {a.iters} iteration(s):")
        for row in grid:
            print(" ".join(f"{v:9.4f}" for v in row))


if __name__ == "__main__":
    main()
