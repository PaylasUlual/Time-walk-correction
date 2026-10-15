#!/usr/bin/env python3
"Checks the intra-pixel position"


import ROOT
import pandas as pd

INPUT_FILE = "simulation/data/output_0deg.root"
DETECTOR_NAME = "lgad_sensor"

# Pixel pitch in mm
PIXEL_PITCH_X = 0.100   
PIXEL_PITCH_Y = 0.250   

f = ROOT.TFile.Open(INPUT_FILE)
tree = f.Get("MCParticle")

rows = []
n_events = tree.GetEntriesFast()
n_no_primary = 0

for event in tree:
    particles = getattr(event, DETECTOR_NAME)  

    primary = None   #Only primary particles are considered to get the hit position of the particle that generates the signal
    for mcp in particles:
        if mcp.isPrimary():
            primary = mcp
            break

    if primary is None:
        n_no_primary += 1
        continue  

    pos = primary.getLocalStartPoint()  #Local entry point in the sensor [mm]
    local_x = pos.X()
    local_y = pos.Y()

    col_frac = (local_x % PIXEL_PITCH_X) / PIXEL_PITCH_X
    row_frac = (local_y % PIXEL_PITCH_Y) / PIXEL_PITCH_Y

    rows.append({
        "local_x": local_x,
        "local_y": local_y,
        "col_frac": col_frac,
        "row_frac": row_frac,
    })

df = pd.DataFrame(rows)

print(f"Entries: {len(df)} (from {n_events} events, "
      f"{n_no_primary} event(s) had no primary particle)")
print()
print(f"local_x range: [{df['local_x'].min():.4f}, {df['local_x'].max():.4f}] mm")
print(f"local_y range: [{df['local_y'].min():.4f}, {df['local_y'].max():.4f}] mm")
print()
print("col_frac summary:")
print(df["col_frac"].describe())
print()
print("row_frac summary:")
print(df["row_frac"].describe())