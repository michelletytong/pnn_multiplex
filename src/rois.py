"""
Manual ROI polygons — the ONLY place a fine-grained brain region enters the pipeline.

You draw the polygons in s07 (napari) and they are stored as GeoJSON in
rois/<image_key>.geojson, one Feature per polygon with properties.name = the ROI
name (e.g. 'NAc_shell'), geometry = Polygon in PIXEL coordinates (x, y).

Hand-drawn elsewhere (QuPath/FIJI) works too — export GeoJSON in pixel coords to
the same path and s07 will pick it up.

Coordinate convention: GeoJSON stores (x, y) = (col, row). napari Shapes store
(row, col). shapes_to_features()/features_to_shapes() do that flip, and are pure
functions so they can be tested without a GUI.
"""
import os
import json
import numpy as np
from matplotlib.path import Path


# ---------------------------------------------------------------- read / assign

def load_polygons(geojson_path):
    """-> list of (name, matplotlib Path). Empty if the file is missing."""
    if not os.path.exists(geojson_path):
        return []
    with open(geojson_path) as f:
        gj = json.load(f)
    out = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        name = props.get("name") or (props.get("classification") or {}).get("name", "")
        geom = feat["geometry"]
        rings = geom["coordinates"]
        if geom["type"] == "Polygon":
            out.append((name, Path(np.asarray(rings[0], float))))
        elif geom["type"] == "MultiPolygon":
            for poly in rings:
                out.append((name, Path(np.asarray(poly[0], float))))
    return out


def polygon_area(path):
    """Absolute area of a matplotlib Path's ring, by the shoelace formula."""
    v = np.asarray(path.vertices, float)
    if len(v) < 3:
        return 0.0
    x, y = v[:, 0], v[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def assign_rois(xy, geojson_path):
    """xy: (N,2) array of (x, y). -> (N,) array of ROI-name strings ('' = outside all).

    THE SMALLEST containing polygon wins. Anatomical regions nest — NAc core sits
    inside the shell — so you draw the full outline of each and the innermost one
    takes the cell. You do not have to draw the shell as a ring with a hole in it,
    and you do not have to subtract anything by hand.

    Resolving by area rather than by file order also makes the result independent of
    the order napari happened to hand back the layers, which it previously was not.
    """
    polys = load_polygons(geojson_path)
    labels = np.array([""] * len(xy), dtype=object)
    if not len(xy) or not polys:
        return labels
    for name, path in sorted(polys, key=lambda np_: polygon_area(np_[1])):
        inside = path.contains_points(xy)
        labels[inside & (labels == "")] = name
    return labels


# ------------------------------------------------------- napari <-> geojson (pure)

def shapes_to_features(named_shapes):
    """[(name, (N,2) array in napari (row, col))] -> list of GeoJSON Features (x, y).

    Polygons with fewer than 3 vertices are dropped (a stray click is not a region).
    Rings are closed explicitly, as GeoJSON requires.
    """
    feats = []
    for name, verts in named_shapes:
        v = np.asarray(verts, float)
        if v.ndim != 2 or v.shape[0] < 3:
            continue
        v = v[:, -2:]                      # tolerate a leading axis from napari
        xy = v[:, ::-1]                    # (row, col) -> (x, y)
        ring = np.vstack([xy, xy[:1]]) if not np.allclose(xy[0], xy[-1]) else xy
        feats.append({
            "type": "Feature",
            "properties": {"name": str(name)},
            "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
        })
    return feats


def features_to_shapes(geojson_path):
    """-> dict roi_name -> list of (N,2) arrays in napari (row, col) order.

    Used by s07 to reload polygons you drew earlier so you can edit rather than redraw.
    """
    out = {}
    if not os.path.exists(geojson_path):
        return out
    with open(geojson_path) as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        name = ((feat.get("properties") or {}).get("name")) or "ROI"
        geom = feat["geometry"]
        rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" \
            else [poly[0] for poly in geom["coordinates"]]
        for ring in rings:
            xy = np.asarray(ring, float)
            out.setdefault(name, []).append(xy[:, ::-1])   # (x, y) -> (row, col)
    return out


def save_geojson(path, features):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=1)
    return path


def roi_pixel_areas(geojson_path, shape, chunk=256):
    """Pixels belonging to each ROI, by the SAME smallest-polygon-wins rule used to
    assign cells. -> {name: n_pixels}.

    This matters for densities: NAc core nests inside NAc shell, so the shell
    POLYGON's area double-counts the core. The area a density is divided by must be
    the area its cells actually came from, so it is rasterised with the identical
    rule rather than computed from the outlines.
    """
    polys = load_polygons(geojson_path)
    if not polys:
        return {}
    H, W = int(shape[0]), int(shape[1])
    counts = {name: 0 for name, _ in polys}
    order = sorted(polys, key=lambda np_: polygon_area(np_[1]))
    xs = np.arange(W) + 0.5
    for r0 in range(0, H, chunk):
        ys = np.arange(r0, min(r0 + chunk, H)) + 0.5
        XX, YY = np.meshgrid(xs, ys)
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        taken = np.zeros(len(pts), bool)
        for name, path in order:
            if taken.all():
                break
            inside = path.contains_points(pts) & ~taken
            counts[name] += int(inside.sum())
            taken |= inside
    return counts


def summarize(geojson_path):
    """'NAc_shell x1, NAc_core x2' — what's stored for one image."""
    polys = load_polygons(geojson_path)
    if not polys:
        return "none"
    counts = {}
    for name, _ in polys:
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(f"{n} x{c}" if c > 1 else n for n, c in counts.items())
