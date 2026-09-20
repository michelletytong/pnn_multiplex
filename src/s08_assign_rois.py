"""
Stage 8 — stamp the hand-drawn ROIs onto the cells.

Rewrites ONLY the `roi` column of each results/<MouseID>_master.csv, by testing each
cell's (x, y) against the polygons in rois/<image_key>.geojson. Cells outside every
polygon get roi='' rather than being pooled into a phantom region.

Safe to re-run after redrawing — nothing else in the master table is touched.

    python src/s08_assign_rois.py
    python src/s08_assign_rois.py --whole-image   # no polygons: region = whole crop
"""
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import config as C
import rois as R
import curate_lib as K


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whole-image", action="store_true",
                    help="for images with no polygon file, treat the WHOLE crop as one "
                         "ROI named after its BrainRegion. Lets you run end to end "
                         "without s06; results group by NAc/HPC instead of subregion.")
    args = ap.parse_args(argv)

    cfg = C.load()
    res = cfg["paths"]["results"]
    masters = sorted(glob.glob(os.path.join(res, "*_master.csv")))
    if not masters:
        sys.exit(f"no <MouseID>_master.csv in {res} — run s06 first.")

    total, assigned, missing_roi, fellback, unknown, areas = 0, 0, [], [], [], []
    for path in masters:
        t = pd.read_csv(path)
        t["roi"] = ""
        for key, idx in t.groupby("image").groups.items():
            geo = os.path.join(cfg["paths"]["rois"], f"{key}.geojson")
            if not os.path.exists(geo):
                if args.whole_image:
                    # whole crop = one region, named by the filename's BrainRegion
                    t.loc[idx, "roi"] = t.loc[idx, "brain_region"]
                    fellback.append(key)
                else:
                    missing_roi.append(key)
                continue
            # effective area, by the same smallest-wins rule as the assignment, so
            # densities are divided by the area their cells actually came from
            lab = K.load_nuclei(cfg, key)
            if lab is not None:
                px = float(cfg.get("pixel_size_um") or 0) or None
                for nm, npx in R.roi_pixel_areas(geo, lab.shape).items():
                    areas.append({"image": key, "roi": nm, "area_px": npx,
                                  "area_um2": npx * px * px if px else np.nan,
                                  "area_mm2": npx * px * px / 1e6 if px else np.nan})
            found = sorted({n for n, _ in R.load_polygons(geo)})
            region = key.rsplit("_", 2)[1] if key.count("_") >= 2 else ""
            _, bad = C.check_roi_names(cfg, region, found)
            if bad:
                unknown.extend((key, n, c) for n, c in bad)
            xy = t.loc[idx, ["x", "y"]].to_numpy(float)
            t.loc[idx, "roi"] = R.assign_rois(xy, geo)
        t.to_csv(path, index=False)

        n, a = len(t), int((t.roi != "").sum())
        total += n
        assigned += a
        counts = t.loc[t.roi != "", "roi"].value_counts().to_dict()
        print(f"{os.path.basename(path)}: {a}/{n} cells assigned"
              + (f"  {counts}" if counts else ""))

    if unknown:
        print(f"\n  [!] {len(unknown)} ROI name(s) in the geojson files are not in the "
              f"study vocabulary (config roi_drawing.vocabulary):")
        for key, n, close in unknown:
            print(f"        {key}: '{n}'" + (f"   close to {close}" if close else ""))
        print("      They were still assigned — but check for a typo before pooling.")
    if fellback:
        print(f"\n  --whole-image: {len(fellback)} image(s) had no polygon, so the whole "
              f"crop was used as one region: {sorted(set(fellback))}"
              f"\n      Results group by BrainRegion, NOT by subregion. Draw polygons "
              f"with `make rois` and re-run to get shell/core, CA1/DG etc.")
    if missing_roi:
        print(f"\n  [!] no ROI file yet for {len(missing_roi)} image(s): "
              f"{sorted(set(missing_roi))}\n      draw them with `make rois`, or pass "
              f"--whole-image to treat each crop as a single region.")
    # areas + per-ROI cell counts, so densities need no extra step
    if areas:
        ar = pd.DataFrame(areas)
        counts = []
        for path in masters:
            t = pd.read_csv(path)
            t["roi"] = t["roi"].fillna("").astype(str)
            g = t[t.roi != ""].groupby(["image", "roi"]).size().rename("n_cells")
            counts.append(g.reset_index())
        if counts:
            ar = ar.merge(pd.concat(counts, ignore_index=True), on=["image", "roi"],
                          how="left")
            ar["n_cells"] = ar["n_cells"].fillna(0).astype(int)
            ar["cells_per_mm2"] = (ar.n_cells / ar.area_mm2).round(1)
        dst = os.path.join(res, "roi_areas.csv")
        ar.to_csv(dst, index=False)
        print(f"\nROI areas -> {dst}")
        print(ar.to_string(index=False))
        print("  (area is the region cells were ASSIGNED from: a nested ROI is "
              "subtracted from the one containing it)")

    print(f"\ntotal: {assigned}/{total} cells inside an ROI")
    if total and assigned == 0:
        print("  [!] NOT ONE cell landed inside a polygon. Check that your ROIs were "
              "drawn on the right images and are in pixel coordinates.")


if __name__ == "__main__":
    main()
