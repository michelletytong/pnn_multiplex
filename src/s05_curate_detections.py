"""
Stage 5 — check and fix every channel's detections in one window.

One napari window per image with all four channels and everything editable:

  DAPI            the nuclei mask, plus 'nuclei: add' / 'nuclei: remove' clicks
                  (same editing as s03 — fix anything you spot now)
  WFA / PNN       the PNN detections
  DARPP32         the DARPP32 objects
  ERa             the ERa objects

Each marker gets its own Points layer, coloured to match its channel. To ADD an
object, pick that marker's layer, use the add-point tool and click. To REMOVE one,
select it and press Delete. Toggle the matching image layer to judge what you see.

Close the window to save:
  results/detections_curated/<key>.csv   every marker's objects, hand-added ones
                                         tagged source='manual'
  results/nuclei_curated/<key>.npy       if you changed any nuclei
Neither overwrites the detector output, so re-running s02/s04 cannot discard your
work; s06 prefers the curated files when they exist.

    python src/s05_curate_detections.py            # images not curated yet
    python src/s05_curate_detections.py --all      # revisit everything
    python src/s05_curate_detections.py M1_NAc_1   # just these
    python src/s05_curate_detections.py --reset M1_NAc_1
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import config as C
import curate_lib as K
import detections as D


def curate_one(cfg, img):
    import napari
    key = img["key"]
    labels = K.load_nuclei(cfg, key)
    n_before = int(len(np.unique(labels[labels > 0])))
    det, was_curated = D.load(cfg, key)
    markers = [m for m in C.detected_markers(cfg) if img.has(m)]
    before = {m: int((det.marker == m).sum()) for m in markers}

    viewer = napari.Viewer(title=f"{key}  —  curate ALL channels, close to save")
    K.add_channel_layers(viewer, cfg, img, visible=("DAPI",))
    lab, add, rem = K.add_nucleus_layers(viewer, labels)

    lays = {}
    for m in markers:
        lays[m] = viewer.add_points(
            D.points(det, m), name=f"{m} objects", size=18,
            face_color="transparent", border_color=K.MARKER_COLORS.get(m, "white"),
            border_width=0.15)
    if markers:
        viewer.layers.selection.active = lays[markers[0]]

    print(f"  {key}: {n_before} nuclei, " +
          ", ".join(f"{m}={before[m]}" for m in markers) +
          ("  (curated earlier)" if was_curated else "  (raw detector output)") +
          "\n    pick a marker layer, add-point tool to ADD, select+Delete to REMOVE"
          "\n    nuclei: click in 'nuclei: add' / 'nuclei: remove'"
          "\n    close the window to save")
    napari.run()

    # nuclei
    edited, counts = K.harvest_nuclei(cfg, lab, add, rem)
    n_after = int(len(np.unique(edited[edited > 0])))
    if n_after != n_before or counts["added"] or counts["removed"]:
        K.save_nuclei(cfg, key, edited)

    # detections: keep the detector's rows where a point is unmoved, tag the rest manual
    out = det.copy()
    after = {}
    for m in markers:
        pts = np.asarray(lays[m].data, float).reshape(-1, 2)
        orig = D.points(det, m)
        rows = []
        for rc in pts:
            j = _match_row(rc, orig)
            if j is None:
                rows.append({"marker": m, "X": rc[1], "Y": rc[0],
                             "score": np.nan, "source": "manual"})
            else:
                src = det[det.marker == m].iloc[j]
                rows.append(src.to_dict())
        new = pd.DataFrame(rows, columns=D.COLUMNS) if rows else \
            pd.DataFrame(columns=D.COLUMNS)
        out = D.replace_marker(out, m, new)
        after[m] = len(new)
    D.save(cfg, key, out, curated=True)

    manual = int((out.source == "manual").sum())
    print(f"    nuclei {n_before} -> {n_after};  " +
          ", ".join(f"{m} {before[m]} -> {after[m]}" for m in markers) +
          f";  {manual} object(s) hand-added in total")
    row = dict(stage="s05_detections", image=key,
               nuclei_before=n_before, nuclei_after=n_after)
    for m in markers:
        row[f"{m}_before"], row[f"{m}_after"] = before[m], after[m]
    K.log(cfg, row)
    return row


def _match_row(rc, orig, tol=0.51):
    """Index of the original point at this position, or None if it is new."""
    if not len(orig):
        return None
    d = np.hypot(orig[:, 0] - rc[0], orig[:, 1] - rc[1])
    j = int(np.argmin(d))
    return j if d[j] <= tol else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--all", action="store_true", help="revisit already-curated images")
    ap.add_argument("--reset", action="store_true", help="discard curation for these keys")
    args = ap.parse_args(argv)

    cfg = C.load()
    images = C.discover(cfg)
    if not images:
        sys.exit(f"no images found in {cfg['paths']['data']}")

    if args.reset:
        for key in (args.keys or list(images)):
            p = D.path_for(cfg, key, curated=True)
            if os.path.exists(p):
                os.remove(p)
                print(f"  removed {p}")
        print("Reset — those images fall back to the raw s04 detections.")
        return

    if args.keys:
        bad = [k for k in args.keys if k not in images]
        if bad:
            sys.exit(f"unknown image key(s): {bad}\nknown: {list(images)}")
        todo = list(args.keys)
    else:
        todo = [k for k in images
                if args.all or not os.path.exists(D.path_for(cfg, k, curated=True))]

    no_nuc = [k for k in todo if K.nuclei_path(cfg, k) is None]
    no_det = [k for k in todo if k not in no_nuc and not D.has_any(cfg, k)]
    if no_nuc:
        print(f"  [!] no nuclei mask for {no_nuc} — run s02; skipping.")
    if no_det:
        print(f"  [!] no detections for {no_det} — run s04; skipping.")
    todo = [k for k in todo if k not in no_nuc and k not in no_det]

    done = [k for k in images if os.path.exists(D.path_for(cfg, k, curated=True))]
    if not todo:
        print(f"All {len(done)}/{len(images)} image(s) curated." if done else "Nothing to do.")
        print("\nNext: `make master`.")
        return

    try:
        import napari  # noqa: F401
    except ImportError:
        sys.exit("napari is not installed in this env.\n"
                 "  pip install 'napari[pyqt5]'\n"
                 "Curation is optional — `make master` works on the raw detections.")

    print(f"{len(todo)} image(s) to curate" + (f", {len(done)} already done" if done else ""))
    rows = []
    for n, key in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {key}")
        rows.append(curate_one(cfg, images[key]))
    if rows:
        print("\n=== summary ===")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nNext: `make master` (it will use the curated detections).")


if __name__ == "__main__":
    main()
