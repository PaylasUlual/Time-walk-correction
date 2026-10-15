#!/usr/bin/env python3
"Generates one detector_{deg}.conf per incidence angle for the full simulation"
import math
import re

ANGLES = [0, 5, 10, 15, 20, 25]
N_EVENTS_PER_ANGLE = 17000 #Gives a total of about 100,000 events
TEMPLATE_PATH = "detector.conf"
OUT_DIR = "configs"


def substitute(pattern, replacement, text, description):
    new_text, n_subs = re.subn(pattern, replacement, text)
    if n_subs == 0:
        raise RuntimeError(
            f"Substitution for '{description}' matched nothing in the "
            f"template. Check that the pattern still matches detector.conf"
        )
    return new_text


with open(TEMPLATE_PATH) as f:
    template = f.read()

for angle in ANGLES:
    rad = math.radians(angle)
    dx, dy, dz = math.sin(rad), 0.0, math.cos(rad) #The beam angle is in the x-z plane only, dy=0

    conf = template

    #Substitute beam direction to the template which is detector.conf
    conf = substitute(
        r"beam_direction\s*=\s*[\d\.\-eE]+\s+[\d\.\-eE]+\s+[\d\.\-eE]+",
        f"beam_direction = {dx:.6f} {dy:.6f} {dz:.6f}",
        conf,
        "beam_direction",
    )

    #Substitute event count for the full simulation
    conf = substitute(
        r"number_of_events\s*=\s*\d+",
        f"number_of_events = {N_EVENTS_PER_ANGLE}",
        conf,
        "number_of_events",
    )

    #Output ROOT file per incidence angle
    conf = substitute(
        r'file_name\s*=\s*"[^"]*output[^"]*\.root"',
        f'file_name = "../data/output_{angle}deg.root"',
        conf,
        "file_name",
    )

    out_path = f"{OUT_DIR}/detector_{angle}deg.conf"
    with open(out_path, "w") as f:
        f.write(conf)

    
    written_events = re.search(r"number_of_events\s*=\s*(\d+)", conf).group(1)
    print(f"wrote {out_path} "
          f"(beam_direction {dx:.3f} {dy:.3f} {dz:.3f}, "
          f"number_of_events={written_events})")