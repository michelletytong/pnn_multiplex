"""
Shared machinery for the two curation GUIs.

  s03_curate_nuclei      the anchor only — nuclei, right after Cellpose
  s05_curate_detections  everything — nuclei plus one editable layer per marker

Both use the same nucleus editing, so a fix made in one behaves identically in the
other. The pure edit functions live here (no napari import at module level) so they
can be tested without a display.
"""
import os
import numpy as np
import pandas as pd

CHANNEL_COLORS = {"DAPI": "blue", "WFA": "green", "DARPP32": "magenta", "ERa": "yellow"}
MARKER_COLORS = {"WFA": "#54A24B", "DARPP32": "#E45756", "ERa": "#EECA3B"}


# ------------------------------------------------------------------ nuclei paths

def nuclei_raw(cfg, key):
    return os.path.join(cfg["paths"]["results"], "nuclei", f"{key}.npy")


def nuclei_curated(cfg, key):
    return os.path.join(cfg["paths"]["results"], "nuclei_curated", f"{key}.npy")


def nuclei_path(cfg, key):
    """The mask s06 would use: curated if it exists, else raw. None if neither."""
    for p in (nuclei_curated(cfg, key), nuclei_raw(cfg, key)):
        if os.path.exists(p):
            return p
    return None


def load_nuclei(cfg, key):
    p = nuclei_path(cfg, key)
    return np.load(p) if p else None


def save_nuclei(cfg, key, labels):
    p = nuclei_curated(cfg, key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.asarray(labels).astype(np.int32))
    return p


# --------------------------------------------------- nucleus editing (pure)

def median_radius(labels):
    """Median equivalent radius (px) of the labelled objects, or None if empty."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    if not len(ids):
        return None
    return float(np.median(np.sqrt(counts / np.pi)))


def apply_nuclei_edits(labels, add_rc=(), remove_rc=(), radius=None):
    """Apply click-to-add / click-to-remove to a label mask.

    Removing takes the whole label under the point. Adding stamps a disc, writing
    only into background so it can never eat an existing nucleus.
    -> (new_labels, n_removed, n_added, n_ignored)
    """
    out = np.asarray(labels).copy()
    H, W = out.shape

    n_removed = 0
    for r, c in remove_rc:
        r, c = int(round(float(r))), int(round(float(c)))
        if 0 <= r < H and 0 <= c < W:
            lab = int(out[r, c])
            if lab:
                out[out == lab] = 0
                n_removed += 1

    if radius is None:
        radius = median_radius(out) or 8.0
    radius = max(1.0, float(radius))

    n_added = n_ignored = 0
    nxt = int(out.max()) + 1
    yy, xx = np.ogrid[:H, :W]
    for r, c in add_rc:
        r, c = int(round(float(r))), int(round(float(c)))
        if not (0 <= r < H and 0 <= c < W):
            n_ignored += 1
            continue
        disc = ((yy - r) ** 2 + (xx - c) ** 2) <= radius ** 2
        free = disc & (out == 0)
        if not free.any():
            n_ignored += 1
            continue
        out[free] = nxt
        nxt += 1
        n_added += 1
    return out, n_removed, n_added, n_ignored


# ------------------------------------------------------------------- logging

def log(cfg, row, name="curation_log.csv"):
    """Append/replace one image's row in an audit log."""
    p = os.path.join(cfg["paths"]["results"], name)
    df = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    if len(df) and "image" in df and "stage" in df:
        df = df[~((df.image == row["image"]) & (df.stage == row["stage"]))]
    elif len(df) and "image" in df:
        df = df[df.image != row["image"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)
    return p


# ----------------------------------------------------------------- napari bits

def add_channel_layers(viewer, cfg, img, visible=("DAPI",)):
    """One image layer per channel present, only `visible` shown at open."""
    order = [r for r in ("DAPI", "WFA", "DARPP32", "ERa") if img.has(r)]
    order += [r for r in img.roles if r not in order]
    for role in order:
        viewer.add_image(_read(img.path(role)), name=role,
                         colormap=CHANNEL_COLORS.get(role, "gray"),
                         blending="additive", visible=role in visible)


def _read(path):
    import io_utils as io
    return io.read(path)


def add_nucleus_layers(viewer, labels):
    """The editable label mask plus the two click layers. -> (labels, add, remove)."""
    lab = viewer.add_labels(np.asarray(labels).astype(np.int32), name="nuclei",
                            opacity=0.45)
    add = viewer.add_points(np.empty((0, 2)), name="nuclei: add", size=10,
                            face_color="lime", border_color="black")
    rem = viewer.add_points(np.empty((0, 2)), name="nuclei: remove", size=10,
                            face_color="red", border_color="black")
    return lab, add, rem


def harvest_nuclei(cfg, labels_layer, add_layer, remove_layer):
    """Apply the clicks to whatever is in the labels layer. -> (mask, counts dict)."""
    radius = (cfg.get("curation") or {}).get("add_nucleus_radius_px")
    edited, n_rm, n_add, n_ign = apply_nuclei_edits(
        np.asarray(labels_layer.data), add_layer.data, remove_layer.data, radius)
    return edited, {"added": n_add, "removed": n_rm, "ignored": n_ign}
