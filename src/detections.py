"""
The detection store — one place for every marker's detected objects.

Every non-anchor marker (WFA/PNN, DARPP32, ERa) is found as OBJECTS in s04 and stored
the same way, whatever detector produced it. One file per image:

    results/detections/<image_key>.csv          raw detector output (s04)
    results/detections_curated/<image_key>.csv  after hand-editing (s05)

columns: marker, X, Y, score, source
    X, Y     object centre in PIXEL coordinates of that image
    score    detector confidence where it exists (Lupori), else blank
    source   'lupori' | 'cellpose' | 'manual'   — so hand-added objects stay visible

s06 prefers the curated file when present, so re-running a detector can never
silently discard your edits. Nothing here knows about intensity or thresholds: an
object either was found or it was not.
"""
import os
import numpy as np
import pandas as pd

COLUMNS = ["marker", "X", "Y", "score", "source"]


# ------------------------------------------------------------------ paths / io

def raw_dir(cfg):
    return os.path.join(cfg["paths"]["results"], "detections")


def curated_dir(cfg):
    return os.path.join(cfg["paths"]["results"], "detections_curated")


def path_for(cfg, key, curated=False):
    return os.path.join(curated_dir(cfg) if curated else raw_dir(cfg), f"{key}.csv")


def has_any(cfg, key):
    """Has ANY detector been run for this image? Distinguishes 'no objects found'
    (a real negative) from 'never looked' (unknown)."""
    return os.path.exists(path_for(cfg, key, True)) or os.path.exists(path_for(cfg, key))


def load(cfg, key, curated=None):
    """-> (DataFrame with COLUMNS, whether it came from the curated file)."""
    cur, raw = path_for(cfg, key, True), path_for(cfg, key, False)
    if curated is True:
        p, is_cur = cur, True
    elif curated is False:
        p, is_cur = raw, False
    else:
        p, is_cur = (cur, True) if os.path.exists(cur) else (raw, False)
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLUMNS), False
    df = pd.read_csv(p)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLUMNS], is_cur


def save(cfg, key, df, curated):
    d = curated_dir(cfg) if curated else raw_dir(cfg)
    os.makedirs(d, exist_ok=True)
    p = path_for(cfg, key, curated)
    df.reindex(columns=COLUMNS).to_csv(p, index=False)
    return p


def points(df, marker):
    """(N,2) array of (row, col) for one marker — napari's order."""
    sub = df[df.marker == marker]
    if not len(sub):
        return np.empty((0, 2), float)
    return sub[["Y", "X"]].to_numpy(float)


def from_points(marker, pts_rc, source="manual", score=np.nan):
    """napari (row, col) points -> detection rows."""
    pts_rc = np.asarray(pts_rc, float).reshape(-1, 2)
    return pd.DataFrame({"marker": marker,
                         "X": pts_rc[:, 1], "Y": pts_rc[:, 0],
                         "score": score, "source": source})


def replace_marker(df, marker, new_rows):
    """Swap one marker's rows, leaving every other marker untouched."""
    keep = df[df.marker != marker] if len(df) else df
    return pd.concat([keep, new_rows], ignore_index=True)


# ------------------------------------------------------------- what was RUN
# Knowing a marker has no rows is not the same as knowing it was never looked for.
# The two detectors need different conda envs, so running one and not the other is
# routine — and without this, s06 would report a confident 0 for a marker whose
# detector never saw the image.

RUN_COLUMNS = ["image", "marker", "detector", "n_objects", "run_at"]


def runs_path(cfg):
    return os.path.join(raw_dir(cfg), "_runs.csv")


def load_runs(cfg):
    p = runs_path(cfg)
    if not os.path.exists(p):
        return pd.DataFrame(columns=RUN_COLUMNS)
    df = pd.read_csv(p)
    for c in RUN_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[RUN_COLUMNS]


def log_run(cfg, key, marker, detector, n):
    """Record that `detector` was run for `marker` on this image, whatever it found."""
    import datetime
    df = load_runs(cfg)
    if len(df):
        df = df[~((df.image == key) & (df.marker == marker))]
    row = {"image": key, "marker": marker, "detector": detector, "n_objects": int(n),
           "run_at": datetime.datetime.now().isoformat(timespec="seconds")}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(raw_dir(cfg), exist_ok=True)
    df.to_csv(runs_path(cfg), index=False)
    return runs_path(cfg)


def assessed_markers(cfg, key, det=None):
    """Markers genuinely assessed for this image -> they may have a 0/1 call.

    From the run log when there is one. Falling back to 'has rows in the detections
    file' is deliberately conservative: a marker that ran and found nothing would be
    reported as unknown rather than as a confident zero.
    """
    runs = load_runs(cfg)
    if len(runs):
        out = set(runs.loc[runs.image == key, "marker"])
        if out:
            return out
    if det is None:
        det, _ = load(cfg, key)
    return set(det.marker.dropna().unique()) if len(det) else set()


# ------------------------------------------------------------------- matching

def match_one_to_one(centroids_xy, det_xy, radius):
    """Assign each detection to at most ONE nucleus, and each nucleus at most one
    detection. Greedy nearest-pair first.

    A perineuronal net wraps a single neuron, and one DARPP32+ soma is one cell. If
    every nucleus within the radius could claim the same object, the positive count
    would exceed the number of objects actually found — which is what happened before
    this existed (23 PNN detections produced 36 PNN+ cells).

    -> (nearest distance per nucleus, assigned bool per nucleus)
    """
    from scipy.spatial import cKDTree
    n = len(centroids_xy)
    nearest = np.full(n, np.nan)
    assigned = np.zeros(n, bool)
    det_xy = np.asarray(det_xy, float).reshape(-1, 2)
    if not n or not len(det_xy):
        return nearest, assigned

    ctree = cKDTree(centroids_xy)
    nearest = cKDTree(det_xy).query(centroids_xy)[0]

    pairs = []
    for di, (px, py) in enumerate(det_xy):
        for ci in ctree.query_ball_point([px, py], radius):
            d = float(np.hypot(centroids_xy[ci][0] - px, centroids_xy[ci][1] - py))
            pairs.append((d, di, ci))
    pairs.sort()
    used_det, used_cell = set(), set()
    for d, di, ci in pairs:
        if di in used_det or ci in used_cell:
            continue
        used_det.add(di)
        used_cell.add(ci)
        assigned[ci] = True
    return nearest, assigned
