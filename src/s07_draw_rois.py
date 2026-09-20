"""
Stage 7 — draw the brain regions of interest, by hand, in napari.

For each image it opens a napari window with:
  - the DAPI and WFA channels as image layers (plus any other markers, hidden)
  - your detected cells overlaid (nuclei as points, PNN+ cells ringed) so you can
    see what you are enclosing  -- config: roi_drawing.show_cells
  - one empty Shapes layer per starter ROI name for that brain region

HOW TO USE
  1. Pick the Shapes layer for the region you want (left panel).
  2. Select the polygon tool and click around the region; press Esc to finish it.
  3. THE LAYER NAME IS THE ROI NAME. Double-click a layer name to rename it, and
     use "New shapes layer" (the + button) to add a region that wasn't predefined.
  4. Close the window to save. The next image opens automatically.

Polygons are written to rois/<image_key>.geojson. Re-running reloads what you drew
so you can adjust it instead of starting over. s07 then stamps the ROI onto cells.

    python src/s08_draw_rois.py              # every image still missing ROIs
    python src/s08_draw_rois.py --all        # revisit images that already have ROIs
    python src/s08_draw_rois.py M1_NAc_1     # just these image keys
"""
import os
import sys
import argparse
import pandas as pd
import config as C
import io_utils as io
import rois as R

# distinct outline colors cycled across ROI layers
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#EECA3B"]


def cells_of(cfg, key):
    """Detected cells for one image, from the master CSVs. -> (nuclei_rc, pnn_rc)."""
    res = cfg["paths"]["results"]
    animal = key.rsplit("_", 2)[0]
    p = os.path.join(res, f"{animal}_master.csv")
    if not os.path.exists(p):
        return None, None
    t = pd.read_csv(p)
    t = t[t.image == key]
    if t.empty:
        return None, None
    nuc = t[["y", "x"]].to_numpy(float)                      # napari wants (row, col)
    pnn = t.loc[t.pos_PNN == 1, ["y", "x"]].to_numpy(float)
    return nuc, pnn


def draw_one(cfg, img, out_path):
    """Open napari for one image; return the features drawn (saved by caller)."""
    import napari

    viewer = napari.Viewer(title=f"{img['key']}  —  draw ROIs, then close to save")

    # image layers: DAPI and WFA visible, anything else available but hidden
    for role in ["DAPI", "WFA"] + [r for r in img.roles if r not in ("DAPI", "WFA")]:
        if not img.has(role):
            continue
        viewer.add_image(io.read(img.path(role)), name=role,
                         colormap={"DAPI": "blue", "WFA": "green"}.get(role, "gray"),
                         blending="additive", visible=role in ("DAPI", "WFA"))

    if (cfg.get("roi_drawing") or {}).get("show_cells", True):
        nuc, pnn = cells_of(cfg, img["key"])
        if nuc is not None and len(nuc):
            viewer.add_points(nuc, name="nuclei", size=6, face_color="white",
                              opacity=0.45, visible=True)
        if pnn is not None and len(pnn):
            viewer.add_points(pnn, name="PNN+ cells", size=16, face_color="transparent",
                              border_color="yellow", border_width=0.15)

    # one Shapes layer per starter ROI name, pre-filled with anything drawn before
    existing = R.features_to_shapes(out_path)
    names = list(existing) or C.roi_layer_names(cfg, img["region"]) or ["ROI"]
    for i, name in enumerate(names):
        color = PALETTE[i % len(PALETTE)]
        viewer.add_shapes(existing.get(name) or None, name=name, shape_type="polygon",
                          edge_color=color, face_color="transparent", edge_width=4)
    viewer.layers.selection.active = viewer.layers[names[0]]

    _key, _ = C.roi_match(cfg, img["region"])
    _via = "" if _key in (None, img["region"]) else f"  (region '{img['region']}' -> "\
                                                    f"vocabulary '{_key}')"
    print(f"  {img['key']}: draw into the Shapes layers, then CLOSE the window to save."
          f"\n    starter layers: {names}{_via}"
          + (f"  (reloaded {sum(len(v) for v in existing.values())} existing polygon(s))"
             if existing else ""))
    napari.run()

    # harvest every Shapes layer present at close — layer name == ROI name
    named = []
    for layer in viewer.layers:
        if type(layer).__name__ != "Shapes":
            continue
        for verts in layer.data:
            named.append((layer.name, verts))
    feats = R.shapes_to_features(named)

    # validate against the study vocabulary before anything is written. Across many
    # images, 'NAc_Shell' vs 'NAc_shell' would otherwise split one region in two
    # with no error anywhere.
    drawn = sorted({f["properties"]["name"] for f in feats})
    _, bad = C.check_roi_names(cfg, img["region"], drawn)
    if bad:
        print(f"    [!] ROI name(s) not in the vocabulary for region "
              f"'{img['region']}':")
        for n, close in bad:
            hint = f"   did you mean {close}?" if close else ""
            print(f"          '{n}'{hint}")
        allowed = C.roi_layer_names(cfg, img["region"])
        print(f"        declared for '{img['region']}': {allowed or '(nothing)'}")
        if C.roi_strict(cfg):
            print(f"        NOT SAVED (roi_drawing.strict: true). Either rename the "
                  f"layer, or add the name under roi_drawing.vocabulary in config.yaml:")
            print(f"            {img['region']}: [{', '.join(sorted(set(allowed) | set(drawn)))}]")
            return None
        print("        saving anyway (roi_drawing.strict: false)")
    return feats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="image keys to draw (default: those missing ROIs)")
    ap.add_argument("--all", action="store_true", help="include images that already have ROIs")
    args = ap.parse_args(argv)

    cfg = C.load()
    images = C.discover(cfg)
    if not images:
        sys.exit(f"no images found in {cfg['paths']['data']}")
    roi_dir = cfg["paths"]["rois"]
    os.makedirs(roi_dir, exist_ok=True)

    def out_path(key):
        return os.path.join(roi_dir, f"{key}.geojson")

    if args.keys:
        unknown = [k for k in args.keys if k not in images]
        if unknown:
            sys.exit(f"unknown image key(s): {unknown}\nknown: {list(images)}")
        todo = list(args.keys)
    else:
        todo = [k for k in images if args.all or not os.path.exists(out_path(k))]

    done = [k for k in images if os.path.exists(out_path(k))]
    if not todo:
        print(f"All {len(images)} image(s) already have ROIs:")
        for k in done:
            print(f"  {k}: {R.summarize(out_path(k))}")
        print("\nRedraw one with:  python src/s08_draw_rois.py <image_key>"
              "\nNext: `make assign`.")
        return

    try:
        import napari     # noqa: F401
    except ImportError:
        sys.exit("napari is not installed in this env.\n"
                 "  conda activate pnn-analysis && pip install 'napari[all]'\n"
                 "Or draw in QuPath/FIJI and export GeoJSON (pixel coords, "
                 f"properties.name = ROI name) to {roi_dir}/<image_key>.geojson")

    undeclared = sorted({images[k]["region"] for k in todo
                         if not C.roi_layer_names(cfg, images[k]["region"])})
    if undeclared:
        print(f"  [!] no ROI vocabulary declared for region(s) {undeclared}. Add them "
              f"under roi_drawing.vocabulary in config.yaml, e.g.")
        for r in undeclared:
            print(f"          {r}: [name_one, name_two]")
        if C.roi_strict(cfg):
            sys.exit("  strict mode: declare them first so names stay consistent "
                     "across images.")

    print(f"{len(todo)} image(s) to draw" + (f", {len(done)} already done" if done else ""))
    for n, key in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {key}")
        feats = draw_one(cfg, images[key], out_path(key))
        if feats is None:
            print(f"  {key}: nothing saved — fix the ROI name(s) above and re-run")
            continue
        if not feats:
            print(f"  no polygons drawn — nothing saved for {key} "
                  f"(it will come up again next run)")
            continue
        R.save_geojson(out_path(key), feats)
        print(f"  saved {len(feats)} polygon(s) -> {out_path(key)}"
              f"  [{R.summarize(out_path(key))}]")

    print("\nNext: `make assign` to stamp these ROIs onto the cells.")


if __name__ == "__main__":
    main()
