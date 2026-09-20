"""
Stage 6 — the master table, ONE CSV PER MOUSE (results/<MouseID>_master.csv).
One row per cell, pooling every brain region and slice for that mouse.

WHAT COUNTS AS A CELL
  Normally a DAPI nucleus. But in a single optical plane a soma can be cut so its
  cytoplasm is visible while its nucleus sits above or below, and that cell would
  then be missing from the table entirely — biasing every density, and unevenly,
  because the effect is worse where the tissue is denser.

  With detection.union_anchor on, a detected object with no nucleus within the match
  radius becomes a cell in its own right. The `anchored_by` column says which:
      DAPI          a segmented nucleus (the normal case)
      DARPP32, ...  a marker object whose nucleus is not in this plane

  An orphan-anchored cell gets NO invented nuclear data. There is no nucleus, so
  nucleus-compartment columns and area_px stay blank for it. What it does carry is
  its location, its ROI, soma-ring measurements taken around the object, and a 1 for
  the marker that anchored it — which is what densities and co-localisation need.
  Exclude them any time with `anchored_by == "DAPI"`.

COLUMNS
  fluoMean_<marker>[_<compartment>]  mean intensity. Suffixed when a marker is
                                     measured in more than one compartment.
  pos_<call>     1 if one of that marker's DETECTED OBJECTS matched this cell, 0 if
                 not, BLANK if no detector ever ran for it on this image
  <marker>_dist  distance in px to the nearest object of that marker
  anchored_by    DAPI, or the marker(s) that anchored an orphan cell

Matching is one-to-one against the FULL anchor set: one object, one cell. There are
no intensity cutoffs anywhere — a pos_ column means an object was detected, never
that a brightness cleared a line.

Inputs prefer hand-curated over raw, so re-running a detector cannot discard edits.
The `roi` column is created EMPTY; draw regions in s07 and s08 fills it in.

    python src/s06_build_master.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops
import config as C
import io_utils as io
import curate_lib as K
import detections as D


def order_columns(cfg, t):
    """Canonical column order, so every <MouseID>_master.csv has the same schema."""
    front = ["image", "cellID", "animal", "brain_region", "slice"]
    ms, dm = list(C.markers(cfg)), C.detected_markers(cfg)
    fluo = [c for m in ms for c in C.fluo_cols(cfg, m)]
    back = (fluo + ["x", "y", "area_px", "anchored_by"]
            + [f"{m}_dist" for m in dm]
            + [f"pos_{C.call_name(cfg, m)}" for m in dm] + ["roi"])
    middle = [c for c in t.columns if c not in front and c not in back]
    return t.reindex(columns=[c for c in front + middle + back
                              if c in t.columns or c in back])


def orphan_anchors(cent_xy, det, markers, radius):
    """Detected objects with no nucleus in range — cells whose nucleus is out of plane.

    Objects of DIFFERENT markers sitting on the same nucleus-less soma are merged
    into one anchor; otherwise a DARPP32+ PNN+ cell with no visible nucleus would
    become two cells. -> (anchor_xy (N,2), [set of markers] per anchor)
    """
    from scipy.spatial import cKDTree
    cand = []
    for m in markers:
        pts = D.points(det, m)[:, ::-1]                 # (row,col) -> (x,y)
        if not len(pts):
            continue
        d = (cKDTree(cent_xy).query(pts)[0] if len(cent_xy)
             else np.full(len(pts), np.inf))
        cand.extend((p, m) for p, dist in zip(pts, d) if dist > radius)
    if not cand:
        return np.empty((0, 2)), []
    anchors, sources = [], []
    for p, m in cand:
        for i, a in enumerate(anchors):
            if np.hypot(a[0] - p[0], a[1] - p[1]) <= radius:
                sources[i].add(m)
                break
        else:
            anchors.append(p)
            sources.append({m})
    return np.array(anchors, float), sources


def score_image(cfg, img, labels, det, assessed):
    """-> DataFrame, one row per cell (nuclei plus, optionally, orphan anchors)."""
    meta = C.meta_of(cfg, img)
    ring_px = cfg["sampling"]["soma_ring_dilate_px"]
    radius = (cfg.get("detection") or {}).get("match_radius_px", 25)
    ms, dm = list(C.markers(cfg)), C.detected_markers(cfg)
    planes = {m: io.channel(cfg, img, m) for m in ms
              if img.has(m) and C.measure_in(cfg, m)}
    H, W = labels.shape

    props = list(regionprops(labels))
    cent = (np.array([[p.centroid[1], p.centroid[0]] for p in props], float)
            if props else np.empty((0, 2)))

    orph_xy, orph_src = ((orphan_anchors(cent, det,
                                         [m for m in dm if m in assessed], radius))
                         if C.union_anchor(cfg) else (np.empty((0, 2)), []))
    if len(orph_xy):
        by = {}
        for s in orph_src:
            k = "+".join(sorted(s))
            by[k] = by.get(k, 0) + 1
        print(f"      union anchor: {len(orph_xy)} cell(s) with no nucleus in plane {by}")

    anchors = np.vstack([cent, orph_xy]) if len(orph_xy) else cent
    n_real = len(cent)

    # match every marker ONCE against the full anchor set, so one-to-one still holds
    matched, dists = {}, {}
    for m in dm:
        if m in assessed:
            dists[m], matched[m] = D.match_one_to_one(
                anchors, D.points(det, m)[:, ::-1], radius)
            print(f"      {m}: {int(matched[m].sum())}/{int((det.marker == m).sum())}"
                  f" object(s) matched a cell within {radius}px")
        else:
            dists[m], matched[m] = np.full(len(anchors), np.nan), None

    r_nuc = K.median_radius(labels) or 8.0
    pad = int(np.ceil(r_nuc)) + ring_px + 2
    rows, n_no_ring = [], 0

    for i in range(len(anchors)):
        xc, yc = anchors[i]
        real = i < n_real
        if real:
            p = props[i]
            r0, c0, r1, c1 = p.bbox
            r0, c0 = max(r0 - ring_px - 1, 0), max(c0 - ring_px - 1, 0)
            r1, c1 = min(r1 + ring_px + 1, H), min(c1 + ring_px + 1, W)
            win = labels[r0:r1, c0:c1]
            mask = win == p.label
            cid, anchored, area = f"{img['key']}_{p.label}", "DAPI", int(p.area)
        else:
            # a synthetic disc ONLY so the ring has a sensible inner edge. It is
            # never read as a nucleus measurement — there is no nucleus here.
            r0, c0 = max(int(yc) - pad, 0), max(int(xc) - pad, 0)
            r1, c1 = min(int(yc) + pad, H), min(int(xc) + pad, W)
            win = labels[r0:r1, c0:c1]
            yy, xx = np.ogrid[r0:r1, c0:c1]
            mask = ((yy - yc) ** 2 + (xx - xc) ** 2) <= r_nuc ** 2
            cid = f"{img['key']}_orphan{i - n_real + 1}"
            anchored, area = "+".join(sorted(orph_src[i - n_real])), np.nan

        rec = dict(image=img["key"], cellID=cid, **meta,
                   x=float(xc), y=float(yc), area_px=area, anchored_by=anchored)

        ring = None
        for m in ms:
            for comp in C.measure_in(cfg, m):
                col = C.fluo_col(cfg, m, comp)
                if m not in planes:
                    rec[col] = np.nan
                    continue
                sub = planes[m][r0:r1, c0:c1]
                if comp == "nucleus":
                    rec[col] = float(sub[mask].mean()) if real else np.nan
                else:
                    if ring is None:
                        ring = ndi.binary_dilation(mask, iterations=ring_px) & (win == 0)
                        if not ring.any():
                            n_no_ring += 1
                    rec[col] = float(sub[ring].mean()) if ring.any() else np.nan

        for m in dm:
            rec[f"{m}_dist"] = (float(dists[m][i])
                                if np.isfinite(dists[m][i]) else np.nan)
            rec[f"pos_{C.call_name(cfg, m)}"] = (
                pd.NA if matched[m] is None else int(matched[m][i]))
        rec["roi"] = ""
        rows.append(rec)

    if n_no_ring:
        print(f"      [!] {n_no_ring} cell(s) hemmed in by neighbours — no background "
              f"left in the ring, so their soma_ring markers are NaN")

    df = pd.DataFrame(rows)
    for m in dm:
        c = f"pos_{C.call_name(cfg, m)}"
        if c in df:
            df[c] = df[c].astype("Int64")
    return df


def main(paths=None):
    cfg = C.load()
    images = C.discover(cfg, paths)
    if not images:
        sys.exit(f"no images found in {cfg['paths']['data']}")

    by_animal, n_cur_nuc, n_cur_det = {}, 0, 0
    for key, img in images.items():
        nuc_path = K.nuclei_path(cfg, key)
        if nuc_path is None:
            print(f"skip {key}: no nuclei mask (run s02)")
            continue
        n_cur_nuc += nuc_path == K.nuclei_curated(cfg, key)
        det, was_cur = D.load(cfg, key)
        n_cur_det += was_cur
        assessed = D.assessed_markers(cfg, key, det)
        never = [m for m in C.detected_markers(cfg) if img.has(m) and m not in assessed]
        if never:
            print(f"  [!] {key}: no detector has run for {never} — pos_ blank "
                  f"(unknown, NOT 0). Run s04 for them.")

        cur = nuc_path == K.nuclei_curated(cfg, key)
        print(f"{key}:" + ("  [curated nuclei]" if cur else "")
              + ("  [curated detections]" if was_cur else ""))
        d = score_image(cfg, img, np.load(nuc_path), det, assessed)
        absent = [m for m in C.markers(cfg) if not img.has(m)]
        n_orph = int((d.anchored_by != "DAPI").sum()) if len(d) else 0
        print(f"      {len(d)} cells ({len(d) - n_orph} DAPI-anchored, "
              f"{n_orph} orphan)" + (f"   (no channel for {absent})" if absent else ""))
        by_animal.setdefault(img["animal"], []).append(d)

    if not by_animal:
        sys.exit("nothing to write — no image had a nuclei mask. Run s02 first.")

    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    for animal, parts in sorted(by_animal.items()):
        t = order_columns(cfg, pd.concat(parts, ignore_index=True))
        dst = os.path.join(cfg["paths"]["results"], f"{animal}_master.csv")
        t.to_csv(dst, index=False)
        print(f"\n{animal}: {len(t)} cells across {t.image.nunique()} image(s) -> {dst}")
    if n_cur_nuc or n_cur_det:
        print(f"\n  used hand-curated nuclei for {n_cur_nuc}/{len(images)} image(s), "
              f"curated detections for {n_cur_det}/{len(images)}")
    print("\nNext: `make rois` to draw your regions, then `make assign`.")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
