#!/bin/bash
set -e
cd "$(dirname "$0")"

for conf in configs/detector_*.conf; do
    echo "=== Running $conf ==="
    allpix -c "$conf"
done

echo "=== All angles done — outputs in data/ ==="
ls -la data/*.root