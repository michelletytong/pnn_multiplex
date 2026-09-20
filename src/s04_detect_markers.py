"""
Stage 4 — detect objects in every non-anchor marker channel.

Each marker in config `markers` declares its detector:
  lupori    WFA -> the pretrained PNN FasterRCNN in counting_perineuronal_nets.
            Must run in the `cpn` env (py3.8 / torch 1.11).
  cellpose  DARPP32, ERa -> Cellpose cpsam segmentation of that channel; each
            object's centroid becomes a detection. Runs in the analysis/seg env.

DETECTIONS ARE FILLED IN PROGRESSIVELY, ACROSS ENVIRONMENTS.
No single env has both Cellpose and Lupori's torch-1.11 stack, so each contributes
the markers it can and leaves the rest alone. Writing one marker never touches
another's rows, and re-running a marker replaces its rows rather than appending.

    conda activate <seg>  && python src/s04_detect_markers.py --detector cellpose
    conda activate cpn    && python src/s04_detect_markers.py --detector lupori

With no --detector it runs whatever THIS env supports and reports what is still
outstanding. --status reports without running anything.

results/detections/_runs.csv records which detector ran on which image, so "ran and
found nothing" (a real 0) stays distinct from "never ran" (blank in the master).

Output: results/detections/<image_key>.csv, one row per object
        (marker, X, Y, score, source). Re-running one detector replaces only that
        marker's rows, so the other markers' detections survive.

Hand-editing happens next, in s05. This stage never writes to detections_curated/.
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import config as C
import io_utils as io
import detections as D

_cp_cache = {}


# ----------------------------------------------------------------- cellpose

def _cellpose_model():
    """Cellpose, tolerating the v3 -> v4 API break; accelerator if torch has one."""
    if "m" in _cp_cache:
        return _cp_cache["m"]
    import cellpose
    from cellpose import models
    ver = str(getattr(cellpose, "version", "?"))
    try:
        import torch
        gpu = bool(torch.cuda.is_available()
                   or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        gpu = False
    if hasattr(models, "Cellpose"):
        model, kwargs = models.Cellpose(model_type="cyto", gpu=gpu), {"channels": [0, 0]}
    else:
        model, kwargs = models.CellposeModel(gpu=gpu, pretrained_model="cpsam"), {}
    print(f"  cellpose v{ver}, gpu={gpu}")
    _cp_cache["m"] = (model, kwargs)
    return _cp_cache["m"]


def detect_cellpose(cfg, img, marker):
    """Segment one channel; each object centroid is a detection.

    Centroids via scipy rather than skimage, so this stage runs in any env that has
    Cellpose — the cellpose env does not necessarily carry scikit-image.
    """
    from scipy import ndimage as ndi
    model, kwargs = _cellpose_model()
    diam = (cfg.get("detection") or {}).get("cellpose_diameter_px")
    min_px = (cfg.get("detection") or {}).get("min_object_px", 0) or 0
    plane = io.channel(cfg, img, marker)
    out = model.eval(plane, diameter=diam, **kwargs)
    masks = np.asarray(out[0] if isinstance(out, tuple) else out).astype(np.int32)

    counts = np.bincount(masks.ravel())
    ids = np.nonzero(counts)[0]
    ids = ids[ids > 0]
    keep = ids[counts[ids] >= min_px]
    n_drop = len(ids) - len(keep)
    if n_drop > 0:
        print(f"      dropped {n_drop} object(s) under {min_px}px")
    if not len(keep):
        return pd.DataFrame(columns=D.COLUMNS)
    cents = ndi.center_of_mass(np.ones(masks.shape, bool), masks, list(keep))
    rc = np.atleast_2d(np.array(cents, float))
    return pd.DataFrame({"marker": marker, "X": rc[:, 1], "Y": rc[:, 0],
                         "score": np.nan, "source": "cellpose"})


# ------------------------------------------------------------------- lupori

def detect_lupori(cfg, images, marker):
    """Shell out to counting_perineuronal_nets/predict.py for every image at once.
    -> dict image_key -> DataFrame."""
    files = {k: im.path(marker) for k, im in images.items() if im.has(marker)}
    if not files:
        print(f"  no {marker} channel anywhere — skipping lupori")
        return {}
    model = cfg["paths"]["pnn_model"]
    if not os.path.isdir(model):
        sys.exit(f"PNN model not found at {model} — see environment/cpn.md")
    tmp = os.path.join(cfg["paths"]["results"], "detections", "_lupori_raw.csv")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    cmd = [sys.executable, "predict.py", model, *sorted(files.values()),
           "-o", os.path.abspath(tmp)]
    print(f"  running the PNN detector on {len(files)} image(s)\n  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cfg["paths"]["cpn_repo"], check=True)

    raw = pd.read_csv(tmp)
    os.remove(tmp)
    out = {}
    for name, sub in raw.groupby("imgName"):
        stem = os.path.splitext(os.path.basename(str(name)))[0]
        d = C.parse_name(cfg, stem)
        if d is None:
            print(f"  [!] '{name}' does not match filename_pattern — skipped")
            continue
        out[d["key"]] = pd.DataFrame({"marker": marker, "X": sub.X.to_numpy(float),
                                      "Y": sub.Y.to_numpy(float),
                                      "score": sub.get("score", np.nan),
                                      "source": "lupori"})
    return out


# --------------------------------------------------------------------- main

def detectors_available():
    """Which detectors this interpreter can actually run. -> (usable set, {why not}).

    A proxy, not a guarantee: it checks importability, not that Lupori's code is
    happy with this torch version.
    """
    import importlib.util as u
    usable, blocked = set(), {}
    if u.find_spec("cellpose"):
        usable.add("cellpose")
    else:
        blocked["cellpose"] = "cellpose not importable"
    miss = [m for m in ("hydra", "omegaconf") if not u.find_spec(m)]
    if miss:
        blocked["lupori"] = f"missing {miss} (needed by predict.py)"
    else:
        usable.add("lupori")
    return usable, blocked


def status(cfg, images):
    """What has been detected for each image, and what is still outstanding."""
    runs = D.load_runs(cfg)
    usable, blocked = detectors_available()
    print(f"this env can run: {sorted(usable) or 'nothing'}"
          + (f"   blocked: {blocked}" if blocked else ""))
    print()
    pending_any = False
    for key, img in images.items():
        done = D.assessed_markers(cfg, key)
        bits = []
        for m in C.detected_markers(cfg):
            if not img.has(m):
                bits.append(f"{m}: no channel")
                continue
            if m in done:
                n = runs.loc[(runs.image == key) & (runs.marker == m), "n_objects"]
                n = int(n.iloc[0]) if len(n) else int((D.load(cfg, key)[0].marker == m).sum())
                bits.append(f"{m}: {n}")
            else:
                bits.append(f"{m}: PENDING ({C.detector_of(cfg, m)})")
                pending_any = True
        print(f"  {key}: " + ",  ".join(bits))
    if pending_any:
        print("\n  run the pending detectors in their env; existing rows are preserved.")
    else:
        print("\n  all markers detected for every image. Next: `make curate`.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["cellpose", "lupori"],
                    help="run only this detector (they need different envs)")
    ap.add_argument("--marker", action="append", help="only this marker (repeatable)")
    ap.add_argument("--status", action="store_true",
                    help="report what is detected and what is outstanding; run nothing")
    args = ap.parse_args(argv)

    cfg = C.load()
    images = C.discover(cfg)
    if not images:
        sys.exit(f"no images found in {cfg['paths']['data']}")

    if args.status:
        return status(cfg, images)

    want = args.marker or C.detected_markers(cfg)
    bad = [m for m in want if m not in C.markers(cfg)]
    if bad:
        sys.exit(f"{bad} are not in config `markers`. Known: {list(C.markers(cfg))}")
    nodet = [m for m in want if not C.detector_of(cfg, m)]
    if nodet:
        print(f"  {nodet} are measured only (no `detect` in config) — nothing to "
              f"detect for them; their intensity is computed in s06.")
        want = [m for m in want if m not in nodet]
    todo = [m for m in want
            if args.detector is None or C.detector_of(cfg, m) == args.detector]
    if not todo:
        sys.exit(f"no marker uses detector '{args.detector}'. "
                 f"markers: { {m: C.detector_of(cfg, m) for m in C.markers(cfg)} }")
    print(f"detecting: { {m: C.detector_of(cfg, m) for m in todo} }")

    # Detections are filled in progressively: one env contributes the cellpose
    # markers, another the lupori ones, and each writes only its own rows. So do
    # what THIS env can and say what is still outstanding -- refusing outright
    # would block the normal workflow. Only an explicit --detector this env cannot
    # run is an error, because that request cannot be honoured at all.
    usable, blocked = detectors_available()
    cannot = [m for m in todo if C.detector_of(cfg, m) not in usable]
    if cannot:
        why = "; ".join(f"{d}: {r}" for d, r in blocked.items())
        if args.detector:
            sys.exit(f"this env cannot run --detector {args.detector} ({why}).\n"
                     f"Run it in the right env: `make detect-cellpose` (seg) or "
                     f"`make detect-lupori` (cpn).")
        print(f"  [~] skipping {cannot} — not runnable here ({why}).")
        print(f"      Their rows and run-log entries are left untouched for the "
              f"other env to fill in.")
        todo = [m for m in todo if m not in cannot]
    if not todo:
        sys.exit("nothing this env can detect. Try the other env.")

    # lupori does every image in one subprocess; cellpose is per image
    lup = {}
    for m in [x for x in todo if C.detector_of(cfg, x) == "lupori"]:
        lup[m] = detect_lupori(cfg, images, m)

    for key, img in images.items():
        got = {}
        for m in todo:
            if not img.has(m):
                continue
            if C.detector_of(cfg, m) == "lupori":
                got[m] = lup.get(m, {}).get(key, pd.DataFrame(columns=D.COLUMNS))
            else:
                got[m] = detect_cellpose(cfg, img, m)
        if not got:
            continue
        existing, _ = D.load(cfg, key, curated=False)
        for m, rows in got.items():
            existing = D.replace_marker(existing, m, rows)
        p = D.save(cfg, key, existing, curated=False)
        # log ONLY after the rows are safely on disk. The other order would let a
        # failed save leave a log claiming the marker was assessed, which is exactly
        # the false-zero this log exists to prevent.
        for m, rows in got.items():
            D.log_run(cfg, key, m, C.detector_of(cfg, m), len(rows))
        counts = {m: len(r) for m, r in got.items()}
        absent = [m for m in todo if not img.has(m)]
        print(f"{key}: {counts}"
              + (f"   (no channel for {absent})" if absent else "")
              + f" -> {os.path.basename(p)}")

    print()
    status(cfg, images)


if __name__ == "__main__":
    main()
