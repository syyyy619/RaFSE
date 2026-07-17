#!/usr/bin/env python3
"""Run RaFSE on the 100K University-1652 gallery."""

from rafse_repro.cli import main

if __name__ == "__main__":
    main(["run", "--gallery-size", "100000", *__import__("sys").argv[1:]])
