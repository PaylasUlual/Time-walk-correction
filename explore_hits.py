#!/usr/bin/env python3
"""
Check of the PixelCharge branches in an allpix-squared ROOT output file.
For each branch prints the entry count, min/max, and number of NaNs.
"""

import uproot
import numpy as np

f = uproot.open("simulation/data/output_0deg.root")
hits = f["PixelCharge"]          

for branch in hits.keys():
    arr = hits[branch].array(library="np")
    print(f"{branch:25s} n={len(arr):6d}  min={np.nanmin(arr):.3g}  "
          f"max={np.nanmax(arr):.3g}  nan={np.isnan(arr.astype(float)).sum() if arr.dtype.kind=='f' else 0}")