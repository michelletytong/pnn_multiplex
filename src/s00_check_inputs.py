"""
Stage 0 — check what the pipeline will actually see in data/, before running anything.

Groups the per-channel TIFFs into images and tells you, per image, which marker each
Channel# resolved to and which markers are absent. Reads nothing but filenames and
headers, writes nothing — run it as often as you like.

Check two things here, because both are silent failures later:
  1. every file you expect appears (a typo in a filename just drops that channel)
  2. Channel# -> marker is right. Pass a file to print its intensity stats; the DAPI
     channel should look obviously different from the others.

    python src/s00_check_inputs.py                     # group data/ into images
    python src/s00_check_inputs.py data/M1_NAc_1_1.tif # stats for specific files
"""
import os
import sys
import config as C
import io_utils as io


def summarize(cfg):
    """Print the image grouping. -> (n_images, n_problems)."""
    images = C.discover(cfg)
    if not images:
        print(f"no images in {cfg['paths']['data']} — expected TIFFs named "
              f"MouseID_BrainRegion_Slice#_Channel#.tif")
        return 0, 1

    roles = list(cfg["channels"].values())
    print(f"channel map: " + ", ".join(f"{n}={r}" for n, r in sorted(cfg["channels"].items())))
    print(f"\n{len(images)} image(s) in {cfg['paths']['data']}:")
    problems = 0
    for k, im in images.items():
        missing = [r for r in roles if r not in im["files"]]
        print(f"  {k}: animal={im['animal']} region={im['region']} "
              f"slice={im['slice']} channels={im.roles}"
              + (f"  (absent: {missing})" if missing else ""))
        if not im.has("DAPI"):
            print(f"      [!] no DAPI — s02 cannot segment nuclei for this image")
            problems += 1
        if not im.has("WFA"):
            print(f"      [!] no WFA — no PNN detection, pos_PNN will be blank")

    animals = sorted({im["animal"] for im in images.values()})
    regions = sorted({im["region"] for im in images.values()})
    print(f"\n{len(animals)} mouse/mice: {animals}")
    print(f"brain regions in filenames: {regions}")

    # every region must have ROI names declared, or s07 will stop later
    for r in regions:
        try:
            key, names = C.roi_match(cfg, r)
        except ValueError as e:
            print(f"  [!] {e}")
            problems += 1
            continue
        if names:
            via = "" if key == r else f"  (via vocabulary entry '{key}')"
            print(f"  ROIs for {r}: {names}{via}")
        else:
            print(f"  [!] no ROI vocabulary for '{r}' — add it under "
                  f"roi_drawing.vocabulary in config.yaml before s07")
            problems += 1

    man = cfg["paths"].get("manifest")
    if man and os.path.exists(man):
        import pandas as pd
        sheet = pd.read_csv(man, dtype=str)
        known = set(sheet["animal"]) if "animal" in sheet else set()
        absent = [a for a in animals if a not in known]
        print(f"samples.csv: {len(known)} mouse/mice"
              + (f"   [!] not listed: {absent}" if absent else "  (all images covered)"))
        if absent:
            problems += 1
    else:
        print("samples.csv: absent (optional) — no condition/sex columns; "
              "run `make samples` to scaffold one")
    return len(images), problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = C.load()
    if argv:
        for p in argv:
            io.describe(p)
        return
    n, problems = summarize(cfg)
    if problems:
        print(f"\n[!] {problems} thing(s) above need attention before running s02.")
    elif n:
        print("\nLooks consistent. Next: `make samples` (optional), then `make detect`.")


if __name__ == "__main__":
    main()
