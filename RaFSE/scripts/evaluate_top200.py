#!/usr/bin/env python3
from rafse_repro.cli import main

if __name__ == "__main__":
    main(["evaluate", *__import__("sys").argv[1:]])

