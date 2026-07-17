#!/usr/bin/env python3
from rafse_repro.cli import main

if __name__ == "__main__":
    main(["build-gallery", *__import__("sys").argv[1:]])

