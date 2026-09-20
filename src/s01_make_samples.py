"""
Stage 1 — generate a starter samples.csv by scanning data/.

animal / brain region / slice all come from the FILENAME, so the sample sheet only
carries what a filename can't: per-animal facts like condition and sex. One row per
mouse, not per image.

    python src/s00_s01_make_samples.py              # write samples.csv (refuses to clobber)
    python src/s00_s01_make_samples.py --force      # overwrite an existing samples.csv
    python src/s00_s01_make_samples.py -o other.csv

Add any extra columns you like — s05 stamps every column onto every cell of that mouse.
There is no region column: brain region comes from the filename and, finer, from the
ROI polygons you draw in s06.
"""
import os
import sys
import argparse
import pandas as pd
import config as C

COLUMNS = ["animal", "condition", "sex"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default=None, help="output CSV (default: paths.manifest)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    args = ap.parse_args(argv)

    cfg = C.load()
    out = args.output or cfg["paths"]["manifest"]
    if os.path.exists(out) and not args.force:
        sys.exit(f"{out} already exists — refusing to overwrite it (pass --force, "
                 f"or -o to write elsewhere).")

    images = C.discover(cfg)
    if not images:
        sys.exit(f"no images found in {cfg['paths']['data']} — expected TIFFs named "
                 f"MouseID_BrainRegion_Slice#_Channel#.tif")

    animals = {}
    for img in images.values():
        a = animals.setdefault(img["animal"], {"images": 0, "regions": set()})
        a["images"] += 1
        a["regions"].add(img["region"])

    df = pd.DataFrame([{"animal": a, "condition": "", "sex": ""}
                       for a in sorted(animals)], columns=COLUMNS)
    df.to_csv(out, index=False)

    print(f"{len(images)} image(s), {len(animals)} mouse/mice:")
    for a in sorted(animals):
        info = animals[a]
        print(f"  {a}: {info['images']} image(s), regions {sorted(info['regions'])}")
    print(f"\nwrote {len(df)} rows -> {out}")
    print("Fill in condition/sex (and add any columns you want) by hand.")


if __name__ == "__main__":
    main()
