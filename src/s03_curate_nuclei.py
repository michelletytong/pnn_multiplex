"""
Stage 3 — curate the DAPI nuclei, before anything else is measured.

This is the anchor step. Every marker is scored per nucleus and every detection is
matched to a nucleus, so a nucleus that is missing here is a cell that is missing
from the whole study, and a spurious one is a phantom cell in every denominator.
Fix it now, once, rather than discovering it downstream.

LAYERS
  DAPI (+ the other channels, hidden — tick them if they help you judge a cell)
  nuclei          the label mask. Paint or erase for fine corrections.
  nuclei: add     click anywhere to add a nucleus (disc of the median radius)
  nuclei: remove  click on a nucleus to delete it whole

Close the window to save. Your edits go to results/nuclei_curated/<key>.npy and
never overwrite the raw Cellpose output, so re-running s02 cannot discard them.

    python src/s03_curate_nuclei.py            # images not curated yet
    python src/s03_curate_nuclei.py --all      # revisit everything
    python src/s03_curate_nuclei.py M1_NAc_1   # just these
    python src/s03_curate_nuclei.py --reset M1_NAc_1
"""
import os
import sys
import argparse
import numpy as np
import config as C
import curate_lib as K


def curate_one(cfg, img):
    import napari
    key = img["key"]
    labels = K.load_nuclei(cfg, key)
    before = int(len(np.unique(labels[labels > 0])))

    viewer = napari.Viewer(title=f"{key}  —  curate NUCLEI, close to save")
    K.add_channel_layers(viewer, cfg, img, visible=("DAPI",))
    lab, add, rem = K.add_nucleus_layers(viewer, labels)
    viewer.layers.selection.active = add

    print(f"  {key}: {before} nuclei"
          f"\n    click in 'nuclei: add' / 'nuclei: remove', or paint the mask directly"
          f"\n    close the window to save")
    napari.run()

    edited, counts = K.harvest_nuclei(cfg, lab, add, rem)
    K.save_nuclei(cfg, key, edited)
    after = int(len(np.unique(edited[edited > 0])))
    if counts["ignored"]:
        print(f"    [!] {counts['ignored']} add click(s) ignored (on an existing "
              f"nucleus, or off-image)")
    print(f"    nuclei {before} -> {after}  (+{counts['added']} -{counts['removed']})")
    row = dict(stage="s03_nuclei", image=key, before=before, after=after,
               added=counts["added"], removed=counts["removed"])
    K.log(cfg, row)
    return row


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
            p = K.nuclei_curated(cfg, key)
            if os.path.exists(p):
                os.remove(p)
                print(f"  removed {p}")
        print("Reset — those images fall back to the raw s02 masks.")
        return

    if args.keys:
        bad = [k for k in args.keys if k not in images]
        if bad:
            sys.exit(f"unknown image key(s): {bad}\nknown: {list(images)}")
        todo = list(args.keys)
    else:
        todo = [k for k in images
                if args.all or not os.path.exists(K.nuclei_curated(cfg, k))]

    missing = [k for k in todo if K.nuclei_path(cfg, k) is None]
    if missing:
        print(f"  [!] no nuclei mask for {missing} — run s02 first; skipping.")
        todo = [k for k in todo if k not in missing]

    done = [k for k in images if os.path.exists(K.nuclei_curated(cfg, k))]
    if not todo:
        print(f"All {len(done)}/{len(images)} image(s) have curated nuclei."
              if done else "Nothing to curate.")
        print("\nNext: `make detect-markers`.")
        return

    try:
        import napari  # noqa: F401
    except ImportError:
        sys.exit("napari is not installed in this env.\n"
                 "  pip install 'napari[pyqt5]'\n"
                 "Curation is optional — `make detect-markers` works on the raw masks.")

    print(f"{len(todo)} image(s) to curate" + (f", {len(done)} already done" if done else ""))
    for n, key in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {key}")
        curate_one(cfg, images[key])
    print("\nNext: `make detect-markers` (it will use the curated nuclei).")


if __name__ == "__main__":
    main()
