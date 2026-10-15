#!/bin/bash

set -e
cd "$(dirname "$0")"       

echo "=== Step 1: generating configs and running allpix-squared ==="
(cd simulation && python3 generate_configs.py && bash run_simulation.sh)

echo "=== Step 2: converting ROOT output to HDF5 ==="
python3 src/convert_to_hdf5.py

echo "=== Step 3: executing analysis notebooks ==="

cd notebooks
for nb in [1-4]-*.ipynb; do
    echo "--- $nb ---"
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done

echo "=== Done. Plots saved to notebooks/plots/ ==="