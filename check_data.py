#!/usr/bin/env python3
"""
Some checks before training
Checks;
    1. Row count, rows per angle and NaN values
    2. Value ranges (col_frac / row_frac in [0,1], charges, cluster size)
    3. Columns that are constant (all rows have the same value)
    4. Time-walk; correlation of delta_t = t_raw - t_true with pixel_charge and with 1/pixel_charge
"""

import h5py
import numpy as np

INPUT_FILE = "data/hits.h5"
EXPECTED_ANGLES = [0, 5, 10, 15, 20, 25]

#Columns written by src/convert_to_hdf5.py
FIELDS = ["t_raw", "pulse_integral", "pixel_charge", "cluster_charge", "cluster_size",
          "col_frac", "row_frac", "angle_deg", "v_bias", "t_true"]


def section(title):
    print()
    print(f"--- {title} " + "-" * max(0, 60 - len(title)))


def main():
    #Load every column into memory and stop if one is missing
    with h5py.File(INPUT_FILE, "r") as f:
        missing = [field for field in FIELDS if field not in f]
        if missing:
            raise SystemExit(f"Missing expected field(s) in {INPUT_FILE}: {missing}")
        data = {field: f[field][:] for field in FIELDS}

    n = len(data["t_raw"])
    print(f"Loaded {INPUT_FILE}: {n} rows")

    # -------------------------------------------------------------------
    section("Per-angle row counts")
    angles_present = np.unique(data["angle_deg"])
    for angle in angles_present:
        count = int(np.sum(data["angle_deg"] == angle))
        print(f"  {angle:>5.1f} deg: {count} rows")

    missing_angles = sorted(set(EXPECTED_ANGLES) - set(angles_present.tolist()))
    if missing_angles:
        print(f"  WARNING: expected angles not present: {missing_angles}")

    # -------------------------------------------------------------------
    section("NaN / missing value check")
    #NaN in t_raw, pulse_integral, pixel_charge, pixel_col/row is expected for events where no pixel fired
    any_nan = False
    for field in FIELDS:
        n_nan = int(np.sum(np.isnan(data[field])))
        if n_nan > 0:
            any_nan = True
            print(f"  WARNING: {field} has {n_nan} NaN value(s) ({100 * n_nan / n:.1f}%)")
    if not any_nan:
        print("  OK: no NaN values in any field")

    # -------------------------------------------------------------------
    section("Field summary statistics")
    header = f"{'field':<14}{'min':>14}{'max':>14}{'mean':>14}{'std':>14}"
    print(header)
    print("-" * len(header))
    for field in FIELDS:
        arr = data[field]
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            print(f"{field:<14}{'all NaN':>14}")
            continue
        print(f"{field:<14}{valid.min():>14.4g}{valid.max():>14.4g}"
              f"{valid.mean():>14.4g}{valid.std():>14.4g}")

    # -------------------------------------------------------------------
    section("Range checks")
    for field in ("col_frac", "row_frac"):
        arr = data[field]
        arr = arr[~np.isnan(arr)]
        out_of_range = np.sum((arr < 0) | (arr > 1))
        if out_of_range > 0:
            print(f"  WARNING: {field} has {out_of_range} value(s) outside [0, 1]")
        else:
            print(f"  OK: {field} within [0, 1]")

    #Should be above digitizer threshold
    for field in ("pixel_charge", "cluster_charge"):
        arr = data[field]
        arr = arr[~np.isnan(arr)]
        print(f"  {field} range: [{arr.min():.1f}, {arr.max():.1f}] e-")

    #Total charge of the cluster can't be smaller than the charge in one of its pixels
    cluster_size = data["cluster_size"]
    if np.any(cluster_size < 1):
        print(f"  WARNING: cluster_size has {int(np.sum(cluster_size < 1))} value(s) < 1")
    if np.any(data["cluster_charge"] < data["pixel_charge"]):
        n_bad = int(np.sum(data["cluster_charge"] < data["pixel_charge"]))
        print(f"  WARNING: {n_bad} row(s) have cluster_charge < pixel_charge "
              f"(cluster sum should be >= any single pixel's charge)")

    # -------------------------------------------------------------------
    section("Degenerate-value check")
    for field in FIELDS:
        arr = data[field]
        valid = arr[~np.isnan(arr)]
        if len(valid) > 1 and np.allclose(valid, valid[0]):
            flag = "  WARNING: " if field not in ("v_bias",) else "  (expected constant) "
            print(f"{flag}{field} is constant across all rows (value={valid[0]:.6g})")

    # -------------------------------------------------------------------
    section("Time-walk check")
    #Time-walk means small pulses cross the threshold later so delta_t should be larger for smaller charges.
    #pixel_charge is used (pixel that fired) instead of cluster_charge because the pulse height of the fired pixel causes the time-walk effect.
    mask = ~(np.isnan(data["t_raw"]) | np.isnan(data["t_true"]) | np.isnan(data["pixel_charge"]))
    charge_v = data["pixel_charge"][mask]
    delta_t = data["t_raw"][mask] - data["t_true"][mask]

    if len(charge_v) < 10:
        print("  Not enough valid rows to check correlation (need >= 10)")
    else:
        r_charge = np.corrcoef(charge_v, delta_t)[0, 1]
        r_inv_charge = np.corrcoef(1.0 / charge_v, delta_t)[0, 1]
        print(f"  correlation(pixel_charge, delta_t)     = {r_charge:+.3f} "
              f"(expect negative)")
        print(f"  correlation(1/pixel_charge, delta_t)   = {r_inv_charge:+.3f} "
              f"(expect positive, should be stronger than the line above)")

        if abs(r_inv_charge) < abs(r_charge):
            print("  NOTE: 1/charge correlation is not stronger than the linear one - "
                  "worth double-checking the CSA/pulse configuration")
        if r_charge >= 0:
            print("  WARNING: charge/delta_t correlation is non-negative - "
                  "time-walk direction looks wrong or physics may not be active")

    section("Done")


if __name__ == "__main__":
    main()