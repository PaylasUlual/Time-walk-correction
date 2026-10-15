#!/usr/bin/env python3
"""
Finds where allpix-squared puts the pixel boundaries in its local coordinates to check if the col_frac formula is correct.
Injects charge at known local x positions (DepositionPointCharge, model = "fixed"), runs allpix once per position, and records which pixel
received the charge. The x positions where the pixel index changes are the real pixel boundaries.

 
Only [DepositionGeant4] is replaced. Every other section of simulation/detector.conf is copied unchanged so the digitizer behaves as in the real simulation.

Note; the col_frac formula tested here (line 164) does not have half-pitch shift. src/convert_to_hdf5.py now includes the shift so this script was used to test the old formula.
"""

import os
import re
import glob
import subprocess

import numpy as np
import ROOT

DETECTOR_TEMPLATE = "simulation/detector.conf"
MADPIX_CONF = "simulation/models/madpix.conf"
DETECTOR_NAME = "lgad_sensor"

DIAG_DIR = "simulation/diag_col_frac"
DIAG_CONF_DIR = "simulation/diag_configs"
DIAG_DATA_DIR = os.path.join(DIAG_DIR, "data")

N_STEPS = 25          #Number of x positions in the sweep
N_CHARGES = 20000     #Charges per position. It is far above the threshold so every position fires.


def parse_pixel_geometry(path=MADPIX_CONF):
    """Reads pixel pitch and the matrix size from madpix.conf"""
    with open(path) as f:
        text = f.read()
    m_size = re.search(r"pixel_size\s*=\s*([\d.]+)um\s+([\d.]+)um", text)
    m_n = re.search(r"number_of_pixels\s*=\s*(\d+)\s+(\d+)", text)
    if not (m_size and m_n):
        raise ValueError(f"Could not parse pixel geometry from {path}")
    pitch_x_mm = float(m_size.group(1)) / 1000.0
    pitch_y_mm = float(m_size.group(2)) / 1000.0
    n_x, n_y = int(m_n.group(1)), int(m_n.group(2))
    return pitch_x_mm, pitch_y_mm, n_x, n_y


def extract_section(text, name):
    m = re.search(rf"\[{re.escape(name)}\].*?(?=\n\[|\Z)", text, flags=re.S)
    if m is None:
        raise ValueError(f"Section [{name}] not found in {DETECTOR_TEMPLATE}")
    return m.group(0)


def build_diag_config(local_x_mm, out_path, root_out_path):
    """
    Writes a config for one scan point.
    It is built from; [Allpix] + [GeometryBuilderGeant4],
    [DepositionPointCharge], the remaining sections of detector.conf,
    and a new [ROOTObjectWriter] with this
    point's output file.
    """
    with open(DETECTOR_TEMPLATE) as f:
        template = f.read()

    #Copied sections unchanged from detector.conf
    keep_sections = []
    for name in ["Allpix", "GeometryBuilderGeant4", "ElectricFieldReader",
                 "GenericPropagation", "PulseTransfer", "CSADigitizer"]:
        try:
            keep_sections.append(extract_section(template, name))
        except ValueError:
            pass  

    #One event per scan point
    allpix_section = keep_sections[0]
    allpix_section = re.sub(r"number_of_events\s*=\s*\d+",
                             "number_of_events = 1", allpix_section)
    keep_sections[0] = allpix_section

    #Put N_CHARGES at the chosen x point. y=0 and z=24 um (middle of the sensor) so the charge is deposited inside the sensor.
    point_charge_section = f"""[DepositionPointCharge]
model = "fixed"
source_type = "mip"
position = {local_x_mm:.6f}mm 0um 24um
number_of_charges = {N_CHARGES}
"""

    writer_section = f"""[ROOTObjectWriter]
file_name = "{os.path.abspath(root_out_path)}"
"""

    conf = "\n\n".join(keep_sections[:2] + [point_charge_section]
                        + keep_sections[2:] + [writer_section])

    with open(out_path, "w") as f:
        f.write(conf)


def run_allpix(conf_path):
    result = subprocess.run(
        ["allpix", "-c", os.path.abspath(conf_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  allpix FAILED for {conf_path}:")
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        return False
    return True


def read_hit_pixel(root_path):
    """Returns (pixel_index_x, pixel_index_y) of the pixel with the most
    charge in this file's single event, or None if nothing was recorded."""
    f = ROOT.TFile.Open(root_path)
    if f is None or f.IsZombie():
        return None
    pc_tree = f.Get("PixelCharge")
    if not isinstance(pc_tree, ROOT.TTree) or pc_tree.GetEntriesFast() == 0:
        f.Close()
        return None
    pc_tree.GetEntry(0)
    pixel_charges = getattr(pc_tree, DETECTOR_NAME)
    if len(pixel_charges) == 0:
        f.Close()
        return None
    best = max(pixel_charges, key=lambda pc: pc.getCharge()) #Take the pixel with the most charge
    idx = best.getPixel().getIndex()
    result = (int(idx.X()), int(idx.Y()))
    f.Close()
    return result


def main():
    os.makedirs(DIAG_CONF_DIR, exist_ok=True)
    os.makedirs(DIAG_DATA_DIR, exist_ok=True)

    pitch_x, pitch_y, n_x, n_y = parse_pixel_geometry()
    print(f"Pixel pitch: {pitch_x*1000:.1f} um x {pitch_y*1000:.1f} um, "
          f"grid: {n_x} x {n_y}")

    #Scan from -0.6 to +0.6 pitch around local_x=0 so at least one pixel to pixel transition is in the range.
    xs = np.linspace(-0.6 * pitch_x, 0.6 * pitch_x, N_STEPS)

    print(f"Sweeping local_x from {xs[0]:.4f} to {xs[-1]:.4f} mm "
          f"({N_STEPS} points)\n")

    records = []
    for local_x in xs:
        tag = f"x{local_x:+.5f}mm"
        conf_path = os.path.join(DIAG_CONF_DIR, f"detector_{tag}.conf")
        root_path = os.path.join(DIAG_DATA_DIR, f"output_{tag}.root")

        build_diag_config(local_x, conf_path, root_path)
        if not run_allpix(conf_path):
            continue

        hit = read_hit_pixel(root_path)
        if hit is None:
            print(f"  local_x={local_x:+.4f} mm: no pixel fired")
            continue


        formula_col_frac = (local_x % pitch_x) / pitch_x
        records.append((local_x, hit[0], formula_col_frac))

    if not records:
        raise SystemExit(
            "No successful runs, every allpix invocation failed or "
            "produced no hit. Check the FATAL/error output above, and "
            "inspect one generated config under "
            f"{DIAG_CONF_DIR}/ against your real detector.conf."
        )

    print(f"{'local_x [mm]':>14}{'hit_pixel_x':>13}{'formula_col_frac':>18}")
    print("-" * 47)
    for local_x, px, cf in records:
        print(f"{local_x:>14.5f}{px:>13d}{cf:>18.4f}")

    #Find where the hit pixel changes for the correct pixel boundaries
    print()
    prev_px = records[0][1]
    boundaries = []
    for i in range(1, len(records)):
        local_x, px, _ = records[i]
        if px != prev_px:
            lo_x = records[i - 1][0]
            hi_x = local_x
            boundaries.append((lo_x, hi_x))
            print(f"Pixel boundary detected between local_x={lo_x:+.5f} mm "
                  f"and {hi_x:+.5f} mm (pixel {prev_px} -> {px})")
            prev_px = px

    if not boundaries:
        print("No pixel transition observed in this sweep range, widen "
              "the sweep (increase the 0.6 factor above) and rerun.")
        return

    print()
    print("Interpretation:")
    print("  The col_frac formula assumes pixel boundaries fall at exact")
    print("  multiples of the pitch, measured from local_x = 0. If the")
    print("  boundary local_x values printed above are NOT close to")
    print("  multiples of the pitch (i.e. not near 0.000, +/-0.100, ...")
    print("  for a 100 um pitch), the formula's assumed origin is wrong")
    print("  and process_file() needs an offset correction. If they ARE")
    print("  close to multiples of the pitch, the formula is correct and")
    print("  the col_frac pileup you saw must come from somewhere else")
    print("  (beam/geometry alignment upstream, or a genuine physics")
    print("  effect), worth re-checking the raw beam local_x")
    print("  distribution directly, independent of col_frac, next.")


if __name__ == "__main__":
    main()