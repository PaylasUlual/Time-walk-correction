#!/usr/bin/env python3
"""Inspects tree/branch structure of a simulation output file."""
import uproot

path = "simulation/data/output_0deg.root"
f = uproot.open(path)

print("=== Non-config keys ===")
for k in f.keys():
    if not k.startswith("config"):
        print(" ", k)

