#!/usr/bin/env python3
"""
Merges the allpix-squared ROOT files (one file per angle) into one HDF5 file which is data/hits.h5 , with one row per simulated event.

pixel_charge is the charge in the main pixel
cluster_charge is the charge summed over all pixels
cluster_size is the number of pixels with charge
pulse_integral is the pulse size
t_raw is the threshold crossing time
pixel_col / pixel_row is the pixel that was fired
local_x / local_y is the true hit position
t_true is the true arrival time

Note on pulse_integral; This is the integral of the pulse with units V*ns and not the peak voltage in mV. It is used as a representation of pulse height.
The 1/amplitude trend of time-walk is not changed.

Note on col_frac / row_frac; In allpix-squared local (0,0) is the centre of a pixel instead of its edge. This is shifted by half a pitch so that
0.5 is the pixel centre and 0 and 1 are the pixel edges.

The trees MCParticle, PixelCharge and PixelHit are read together. Entry i in each tree is the same event.

"""

import os
import glob
import re

import h5py
import numpy as np
import ROOT

#Configuration

INPUT_DIR = "simulation/data"
INPUT_GLOB = os.path.join(INPUT_DIR, "output_*deg.root")
OUTPUT_FILE = "data/hits.h5"

DETECTOR_NAME = "lgad_sensor"

#Pixel pitch and bias voltage are read from the simulation config files
MADPIX_CONF = "simulation/models/madpix.conf"
DETECTOR_CONF = "simulation/detector.conf"

NS_TO_PS = 1000.0 #allpix uses ns, this is converted to ps


#Config parsing

def parse_pixel_pitch(path=MADPIX_CONF):
    """Reads pixel_size in um from madpix.conf and returns (pitch_x, pitch_y) in mm."""
    with open(path) as f:
        text = f.read()
    match = re.search(r"pixel_size\s*=\s*([\d.]+)um\s+([\d.]+)um", text)
    if match is None:
        raise ValueError(f"Could not parse pixel_size from {path}")
    pitch_x_um, pitch_y_um = float(match.group(1)), float(match.group(2))
    return pitch_x_um / 1000.0, pitch_y_um / 1000.0  


def parse_bias_voltage(path=DETECTOR_CONF):
    """Reads the operating bias voltage V from detector.conf."""
    with open(path) as f:
        text = f.read()
    match = re.search(r"voltage\s*=\s*([\d.]+)V", text)
    if match is None:
        raise ValueError(f"Could not parse voltage from {path}")
    return float(match.group(1))


PIXEL_PITCH_X, PIXEL_PITCH_Y = parse_pixel_pitch()
V_BIAS = parse_bias_voltage()


#Helpers

def angle_from_filename(path):
    """Gets the incidence angle in degrees from files like output_15deg.root."""
    match = re.search(r"output_(\d+)deg\.root", os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse angle from filename: {path}")
    return float(match.group(1))


def index_pixel_charges(pixel_charges):
    """Builds a lookup {(col, row): PixelCharge} for one event."""
    lookup = {}
    for pc in pixel_charges:
        idx = pc.getPixel().getIndex()
        lookup[(int(idx.X()), int(idx.Y()))] = pc
    return lookup


def process_file(path, angle_deg, rows):
    """Reads one angle's ROOT file and adds one record per event to 'rows'."""
    f = ROOT.TFile.Open(path)
    if f is None or f.IsZombie():
        print(f"  WARNING: could not open {path}, skipping")
        return

    mcp_tree = f.Get("MCParticle")
    pc_tree = f.Get("PixelCharge")
    hit_tree = f.Get("PixelHit")

    if mcp_tree is None or pc_tree is None:
        print(f"  WARNING: missing MCParticle/PixelCharge tree in {path}, skipping")
        f.Close()
        return

    #f.Get() can return a placeholder TObject instead of None when the tree is missing so check the type
    have_hits = isinstance(hit_tree, ROOT.TTree)
    if not have_hits:
        print(f"  WARNING: no PixelHit tree in {path} "
              f"(no hits crossed threshold) - pulse_integral/t_raw will be NaN")

    n_entries = mcp_tree.GetEntriesFast()
    n_written = 0
    n_multi_primary = 0

    for i in range(n_entries):
        mcp_tree.GetEntry(i)
        pc_tree.GetEntry(i)
        if have_hits:
            hit_tree.GetEntry(i)

        particles = getattr(mcp_tree, DETECTOR_NAME)
        if len(particles) == 0:
            continue  #No primary particle recorded for this event

        #Use the primary particle for the true position and time. An event can have more than one primary-flagged particle and they are not sorted by time so take the earlier one.
        mcp = None
        n_primary_here = 0
        for p in particles:
            if p.isPrimary():
                n_primary_here += 1
                if mcp is None or p.getGlobalTime() < mcp.getGlobalTime():
                    mcp = p
        #This check is outside the loop to prevent the count being wrong
        if n_primary_here > 1:
            n_multi_primary += 1
        if mcp is None:
            continue  #No primary particle recorded for this event

        pos = mcp.getLocalStartPoint()
        local_x, local_y = pos.X(), pos.Y()
        #Shift by half a pitch because (0,0) is the pixel centre
        col_frac = ((local_x + PIXEL_PITCH_X / 2.0) % PIXEL_PITCH_X) / PIXEL_PITCH_X
        row_frac = ((local_y + PIXEL_PITCH_Y / 2.0) % PIXEL_PITCH_Y) / PIXEL_PITCH_Y
        t_true = mcp.getGlobalTime() * NS_TO_PS

        pixel_charges = getattr(pc_tree, DETECTOR_NAME)
        charge_lookup = index_pixel_charges(pixel_charges)
        #Charge and size of the whole cluster
        cluster_charge = sum(pc.getCharge() for pc in pixel_charges)
        cluster_size = len(pixel_charges)

        if have_hits:
            pixel_hits = getattr(hit_tree, DETECTOR_NAME)
        else:
            pixel_hits = []

        if len(pixel_hits) > 0:
            #Write one row per event. The fired pixel with the most charge should be chosen.
            #The neighbouring pixels with shared charge are fired late so they would disturb the time-walk fit.
            hit_lookup = {}
            for hit in pixel_hits:
                idx = hit.getPixel().getIndex()
                hit_lookup[(int(idx.X()), int(idx.Y()))] = hit

            #Only use pixels that fired and have a charge entry. If none have it, the pixel with the largest signal should be used.
            fired_with_charge = [k for k in hit_lookup if k in charge_lookup]
            if fired_with_charge:
                primary_key = max(fired_with_charge,
                                   key=lambda k: charge_lookup[k].getCharge())
            else:
                primary_key = max(hit_lookup,
                                   key=lambda k: hit_lookup[k].getSignal())

            hit = hit_lookup[primary_key]
            pc = charge_lookup.get(primary_key)
            pixel_charge = pc.getCharge() if pc is not None else np.nan

            rows.append({
                "t_raw": hit.getGlobalTime() * NS_TO_PS,
                "pulse_integral": hit.getSignal(),
                "pixel_charge": pixel_charge,
                "cluster_charge": cluster_charge,
                "cluster_size": cluster_size,
                "col_frac": col_frac,
                "row_frac": row_frac,
                "pixel_col": primary_key[0],
                "pixel_row": primary_key[1],
                "angle_deg": angle_deg,
                "v_bias": V_BIAS,
                "t_true": t_true,
            })
            n_written += 1
        elif cluster_size > 0:
            #Charge was deposited but no pixels fired. The event is kept so the charge is not lost.
            #The digitizer columns, pixel_col and pixel_row are NaN so they must be removed before training.
            rows.append({
                "t_raw": np.nan,
                "pulse_integral": np.nan,
                "pixel_charge": np.nan,
                "cluster_charge": cluster_charge,
                "cluster_size": cluster_size,
                "col_frac": col_frac,
                "row_frac": row_frac,
                "pixel_col": np.nan,
                "pixel_row": np.nan,
                "angle_deg": angle_deg,
                "v_bias": V_BIAS,
                "t_true": t_true,
            })
            n_written += 1

    f.Close()
    print(f"  {os.path.basename(path)}: {n_written} rows from {n_entries} events "
          f"({n_multi_primary} event(s) had multiple primary-flagged MCParticles)")


def apply_t_true_outlier_cut(data, n_mad=8):
    """Removes rows with an unphysical t_true such as rare hard-scattering events that delay the primary before it reaches the sensor.

    The cutoff is meadian + n_mad*robust_std. Mean + n*std is not used because the outliers would make the std too large and the cutoff too loose.
    """
    t_true = data["t_true"]
    median = np.median(t_true)
    mad = np.median(np.abs(t_true - median))
    robust_std = 1.4826 * mad  #Makes mad comparable to std for a normal distribution
    cutoff = median + n_mad * robust_std

    mask = t_true <= cutoff
    n_removed = int(np.sum(~mask))

    print(f"t_true outlier cut: removing {n_removed} row(s) "
          f"({100 * n_removed / len(t_true):.3f}%) with t_true > {cutoff:.2f} ps "
          f"(median={median:.2f} ps, robust_std={robust_std:.2f} ps, n_mad={n_mad})")

    return {field: arr[mask] for field, arr in data.items()}


# Main

def main():
    files = sorted(glob.glob(INPUT_GLOB), key=angle_from_filename)
    if not files:
        raise SystemExit(f"No input files found matching {INPUT_GLOB}")

    rows = []
    print(f"Converting {len(files)} file(s):")
    for path in files:
        angle_deg = angle_from_filename(path)
        print(f"Processing {path} (angle={angle_deg} deg)")
        process_file(path, angle_deg, rows)

    if not rows:
        raise SystemExit("No hits were collected across any input file")

    fields = ["t_raw", "pulse_integral", "pixel_charge", "cluster_charge", "cluster_size",
              "col_frac", "row_frac", "pixel_col", "pixel_row",
              "angle_deg", "v_bias", "t_true"]
    data = {field: np.array([r[field] for r in rows], dtype=np.float64)
            for field in fields}

    data = apply_t_true_outlier_cut(data, n_mad=8)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with h5py.File(OUTPUT_FILE, "w") as hf:
        for field, arr in data.items():
            hf.create_dataset(field, data=arr)

    n_valid = np.sum(~np.isnan(data["t_raw"]))
    print()
    print(f"Wrote {len(rows)} total rows to {OUTPUT_FILE}")
    print(f"  of which {n_valid} have valid t_raw/pulse_integral (digitized hits)")
    print(f"  and {len(rows) - n_valid} are charge-only rows (no PixelHit available)")


if __name__ == "__main__":
    main()