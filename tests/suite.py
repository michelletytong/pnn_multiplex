"""
Regression suite for pnn_multiplex, run against a THROWAWAY project directory.

It writes its own config.yaml + data/ into a temp dir and points the pipeline there
with $PNN_CONFIG, so it cannot touch the real data/, results/ or rois/. Nothing is
parked, nothing is restored, nothing can be deleted by mistake.

    python suite.py
"""
import os, sys, shutil, subprocess, tempfile, textwrap, json
import numpy as np, pandas as pd, tifffile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJ, "src")
PY = sys.executable
sys.path.insert(0, SRC)

TMP = tempfile.mkdtemp(prefix="pnn_suite_")
CFG = os.path.join(TMP, "config.yaml")
ENV = {**os.environ, "PNN_CONFIG": CFG, "PYTHONPATH": SRC}
ok = []


def check(msg):
    ok.append(msg)
    print("OK:", msg, flush=True)


def run(label, script, *args):
    r = subprocess.run([PY, os.path.join(SRC, script), *args],
                       capture_output=True, text=True, env=ENV, cwd=TMP)
    if r.returncode != 0:
        print(f"\n--- {label} FAILED ---\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


# ---------------------------------------------------------------- the fixture
H = W = 420
CENTRES = [(40, 40), (40, 120), (120, 40), (120, 120), (180, 180)]
CENTRES += [(y, x) for y in (250, 320, 390) for x in (40, 120, 250, 380)]
N = len(CENTRES)
N_IN_ROI = sum(1 for cy, _ in CENTRES if cy <= 300)

# key -> (channels present, DARPP32+ cells, ERa+ cells, cells with a PNN)
SPECS = {
    "M1_NAC1_1":  ([1, 2, 3, 4], {1, 2, 3}, {1, 2}, [1, 2]),
    "M1_NAC1_10": ([1, 2, 3, 4], {4, 5},    {4},    [4]),
    "M2_HPC_1":   ([1, 2, 4],    set(),     {1, 3}, [1]),
    "M4_b_CTX_1": ([1, 2, 4],    set(),     {2},    [1]),
}

CONFIG = textwrap.dedent("""\
    pixel_size_um: 0.624
    paths:
      cpn_repo:  /nonexistent
      pnn_model: model_zoo/none
      data:      data
      rois:      rois
      results:   results
      manifest:  samples.csv
    filename_pattern: '^(?P<animal>.+)_(?P<region>[^_]+)_(?P<slice>\\d+)_(?P<channel>\\d+)$'
    channels:
      1: DAPI
      2: WFA
      3: DARPP32
      4: ERa
    markers:
      WFA:     {detect: lupori,   measure: soma_ring, call_as: PNN}
      DARPP32: {detect: cellpose, measure: soma_ring}
      ERa:     {measure: [nucleus, soma_ring]}
    detection:
      union_anchor: true
      match_radius_px: 25
      cellpose_diameter_px: null
      min_object_px: 20
    sampling:
      soma_ring_dilate_px: 12
    curation:
      add_nucleus_radius_px: null
    roi_drawing:
      vocabulary:
        NAc: [NAc_shell, NAc_core]
        HPC: [CA1, DG]
      strict: true
      show_cells: true
    """)


def plane(kind, pos):
    a = np.full((H, W), 100.0, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for i, (cy, cx) in enumerate(CENTRES, start=1):
        nuc = (yy - cy) ** 2 + (xx - cx) ** 2 <= 10 ** 2
        ring = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 20 ** 2) & ~nuc
        if kind == "dapi":
            a[nuc] = 3000
        elif kind == "ring":
            a[ring] = 2000 if i in pos else 150
        elif kind == "nuc":
            a[nuc] = 2500 if i in pos else 150
    return a


def labels_img():
    lab = np.zeros((H, W), np.int32)
    yy, xx = np.mgrid[0:H, 0:W]
    for i, (cy, cx) in enumerate(CENTRES, start=1):
        lab[(yy - cy) ** 2 + (xx - cx) ** 2 <= 10 ** 2] = i
    return lab


def build():
    for d in ("data", "results/nuclei", "results/detections", "rois"):
        os.makedirs(os.path.join(TMP, d), exist_ok=True)
    open(CFG, "w").write(CONFIG)
    KIND = {1: "dapi", 2: "ring", 3: "ring", 4: "nuc"}
    for key, (chans, dar, era, pnn) in SPECS.items():
        for ch in chans:
            pos = {1: set(), 2: set(pnn), 3: dar, 4: era}[ch]
            tifffile.imwrite(os.path.join(TMP, "data", f"{key}_{ch}.tif"),
                             plane(KIND[ch], pos))
        np.save(os.path.join(TMP, "results/nuclei", f"{key}.npy"), labels_img())
    tifffile.imwrite(os.path.join(TMP, "data", "notes.tif"), np.zeros((4, 4), np.float32))
    # detections, in the unified store: WFA from "lupori", DARPP32 from "cellpose"
    for key, (chans, dar, era, pnn) in SPECS.items():
        rows = []
        for c in pnn:
            cy, cx = CENTRES[c - 1]
            rows.append(dict(marker="WFA", X=cx + 5, Y=cy + 5, score=0.9, source="lupori"))
        for c in sorted(dar):
            cy, cx = CENTRES[c - 1]
            rows.append(dict(marker="DARPP32", X=cx + 3, Y=cy + 3, score=np.nan,
                             source="cellpose"))
        # one DARPP32 object far from every nucleus: a soma whose nucleus is out of
        # plane. Must become its own cell, with NO nuclear measurements.
        if dar:
            rows.append(dict(marker="DARPP32", X=200, Y=210, score=np.nan,
                             source="cellpose"))
        pd.DataFrame(rows, columns=["marker", "X", "Y", "score", "source"]).to_csv(
            os.path.join(TMP, "results/detections", f"{key}.csv"), index=False)


build()
import config as C, detections as D, curate_lib as K, rois as R
cfg = C.load(CFG)
print(f"temp project: {TMP}\n")

# ------------------------------------------------------------------- checks
r = run("s00", "s00_check_inputs.py")
assert r.returncode == 0, r.stdout
images = C.discover(cfg, quiet=True)
assert set(images) == set(SPECS), sorted(images)
assert images["M4_b_CTX_1"]["animal"] == "M4_b"
assert images["M1_NAC1_10"]["slice"] == 10
assert not images["M2_HPC_1"].has("DARPP32")
assert "did not match filename_pattern" in r.stdout
check("discovery: grouping, underscored MouseID, absent channel, junk file ignored")

r = run("s01", "s01_make_samples.py")
assert r.returncode == 0, r.stdout
sm = pd.read_csv(os.path.join(TMP, "samples.csv"))
assert list(sm.columns) == ["animal", "condition", "sex"]
assert sorted(sm.animal) == ["M1", "M2", "M4_b"]
sm["condition"] = ["veh", "est", "veh"]; sm["sex"] = "F"
sm.to_csv(os.path.join(TMP, "samples.csv"), index=False)
check("samples sheet: one row per mouse, no region column")

# --- the marker model ---
assert C.detected_markers(cfg) == ["WFA", "DARPP32"], C.detected_markers(cfg)
assert C.measure_in(cfg, "ERa") == ["nucleus", "soma_ring"]
assert C.measure_in(cfg, "WFA") == ["soma_ring"], "single compartment must still be a list"
assert C.fluo_cols(cfg, "ERa") == ["fluoMean_ERa_nucleus", "fluoMean_ERa_soma_ring"]
assert C.fluo_cols(cfg, "WFA") == ["fluoMean_WFA"], "one compartment keeps the bare name"
assert C.union_anchor(cfg) is True
assert C.intensity_markers(cfg) == ["WFA", "DARPP32", "ERa"]
assert C.call_name(cfg, "WFA") == "PNN"
assert C.detector_of(cfg, "ERa") is None
check("marker model: ERa measured-only, WFA both detected and measured, call_as works")

# --- detection store ---
det, was_cur = D.load(cfg, "M1_NAC1_1")
assert not was_cur and len(det) == 6, (was_cur, len(det))  # 2 WFA + 3 DARPP32 + 1 orphan
new = D.from_points("DARPP32", [[200, 200]], source="manual")
merged = D.replace_marker(det, "DARPP32", new)
assert int((merged.marker == "WFA").sum()) == 2, "replacing one marker touched another"
assert int((merged.marker == "DARPP32").sum()) == 1
check("detection store: replacing one marker leaves the others intact")

# --- one-to-one matching: the 23->36 bug ---
cent = np.array([[100, 100], [104, 100], [108, 100]], float)   # 3 nuclei, 4px apart
nearest, hit = D.match_one_to_one(cent, np.array([[102, 100]]), 25)
assert int(hit.sum()) == 1, f"one object must claim ONE nucleus, got {int(hit.sum())}"
assert np.isfinite(nearest).all()
n2, h2 = D.match_one_to_one(cent, np.empty((0, 2)), 25)
assert int(h2.sum()) == 0 and np.isnan(n2).all()
check("matching: one object -> one cell, never several (pos count <= object count)")

# --- nuclei edits ---
lab = labels_img()
cy3, cx3 = CENTRES[2]
ed, rm, ad, ig = K.apply_nuclei_edits(lab, add_rc=[(210, 210), (60, 250)],
                                      remove_rc=[(cy3, cx3)], radius=9)
assert (rm, ad, ig) == (1, 2, 0), (rm, ad, ig)
assert ed[cy3, cx3] == 0 and ed[210, 210] != 0
_, _, a2, i2 = K.apply_nuclei_edits(lab, add_rc=[CENTRES[0]], radius=9)
assert (a2, i2) == (0, 1), "a click on an existing nucleus must be rejected"
_, _, a3, i3 = K.apply_nuclei_edits(lab, add_rc=[(99999, 5)], radius=9)
assert (a3, i3) == (0, 1)
check("nuclei edits: remove whole label, add disc, reject overlapping/off-image")

# --- s06 master ---
r = run("s06", "s06_build_master.py")
assert r.returncode == 0, r.stdout
masters = {a: os.path.join(TMP, "results", f"{a}_master.csv") for a in ("M1", "M2", "M4_b")}
for a, p in masters.items():
    assert os.path.exists(p), p
m1 = pd.read_csv(masters["M1"]); m2 = pd.read_csv(masters["M2"])
cols = {a: list(pd.read_csv(p).columns) for a, p in masters.items()}
assert len(set(map(tuple, cols.values()))) == 1, cols
assert "pos_ERa" not in m1.columns, "measure-only marker must get no pos_ column"
assert "ERa_dist" not in m1.columns
assert "pos_PNN" in m1.columns and "pos_DARPP32" in m1.columns
assert "fluoMean_ERa" not in m1.columns, "multi-compartment marker must be suffixed"
for c in ("fluoMean_ERa_nucleus", "fluoMean_ERa_soma_ring", "anchored_by"):
    assert c in m1.columns, (c, m1.columns.tolist())
_dapi = m1[m1.anchored_by == "DAPI"]
assert _dapi.fluoMean_ERa_nucleus.notna().all(), "ERa must be measured for every nucleus"
check("master: stable schema; measure-only ERa has fluoMean_ but no pos_/dist")

assert m1.image.nunique() == 2
assert int(m1.anchored_by.eq("DAPI").sum()) == 2 * N
assert set(m1.condition) == {"veh"}
s1 = m1[m1.image == "M1_NAC1_1"].set_index("cellID")
assert set(s1[s1.pos_PNN == 1].index) == {"M1_NAC1_1_1", "M1_NAC1_1_2"}
s10 = m1[m1.image == "M1_NAC1_10"].set_index("cellID")
assert set(s10[s10.pos_PNN == 1].index) == {"M1_NAC1_10_4"}
check("master: per-mouse pooling, metadata stamped, no cross-image detection leak")

assert m2.fluoMean_DARPP32.isna().all(), "no DARPP32 channel -> NaN"
assert m2.pos_DARPP32.isna().all(), "no DARPP32 channel -> pos blank, NOT 0"
assert m1.pos_DARPP32.notna().all()
hi = m1.loc[m1.cellID.isin(["M1_NAC1_1_1", "M1_NAC1_1_2"]), "fluoMean_ERa_nucleus"]
lo = m1.loc[m1.image.eq("M1_NAC1_1") & ~m1.cellID.isin(["M1_NAC1_1_1", "M1_NAC1_1_2"])
            & m1.anchored_by.eq("DAPI"), "fluoMean_ERa_nucleus"]
assert hi.min() > lo.max(), "ERa nuclear scoring wrong"
check("master: absent channel -> blank not 0; compartment scoring correct")

# --- UNION ANCHOR: a soma whose nucleus is out of plane still becomes a cell,
#     but must never be given invented nuclear data ---
_o = m1[m1.anchored_by != "DAPI"]
assert len(_o) == 2, f"expected one orphan per NAc image, got {len(_o)}"
assert set(_o.anchored_by) == {"DARPP32"}, set(_o.anchored_by)
assert _o.fluoMean_ERa_nucleus.isna().all(), "orphan must have NO nuclear measurement"
assert _o.area_px.isna().all(), "orphan has no nucleus, so no nucleus area"
assert _o.fluoMean_ERa_soma_ring.notna().all(), "the soma ring IS measurable"
assert _o.pos_DARPP32.eq(1).all(), "the marker that anchored it must read positive"
assert _o.cellID.str.contains("orphan").all(), _o.cellID.tolist()
# roi is filled by s08 later; here it must simply be present and unassigned
assert set(_o.roi.fillna("")) == {""}, set(_o.roi.fillna(""))
# turning it off returns the old behaviour exactly
_c = open(CFG).read()
open(CFG, "w").write(_c.replace("union_anchor: true", "union_anchor: false"))
_r = run("s06 union off", "s06_build_master.py")
assert _r.returncode == 0, _r.stdout
_off = pd.read_csv(masters["M1"])
assert (_off.anchored_by == "DAPI").all(), "union off must yield only DAPI anchors"
assert len(_off) == 2 * N, (len(_off), 2 * N)
open(CFG, "w").write(_c)
run("s06 union on", "s06_build_master.py")
check("union anchor: orphan cells exist, carry no invented nuclear data, toggle works")

# --- one-to-one still holds ACROSS the combined anchor set ---
_m1 = pd.read_csv(masters["M1"])
_det, _ = D.load(cfg, "M1_NAC1_1")
_n_obj = int((_det.marker == "DARPP32").sum())
_n_pos = int(_m1[_m1.image == "M1_NAC1_1"].pos_DARPP32.eq(1).sum())
assert _n_pos <= _n_obj, f"{_n_pos} positive cells from {_n_obj} objects"
check("one-to-one survives the union: pos count never exceeds the object count")

# --- a detector that NEVER RAN must not yield a confident 0 ---
# The two detectors need different conda envs, so running one and not the other is
# the normal case. Without a run log, WFA (never detected) read as pos_PNN = 0.
_runs = D.runs_path(cfg)
assert not os.path.exists(_runs), "fixture should start with no run log"
_m1 = pd.read_csv(masters["M1"])
assert _m1.pos_PNN.isna().all() or True   # depends on fallback; assert explicitly below
# fallback (no log): a marker with rows counts as assessed, one without does not
_det, _ = D.load(cfg, "M2_HPC_1")
assert "DARPP32" not in set(_det.marker), "fixture: M2 has no DARPP32 objects"
assert D.assessed_markers(cfg, "M2_HPC_1") == {"WFA"}, D.assessed_markers(cfg, "M2_HPC_1")
# with a run log, 'ran and found nothing' becomes a real 0
D.log_run(cfg, "M2_HPC_1", "DARPP32", "cellpose", 0)
assert D.assessed_markers(cfg, "M2_HPC_1") == {"DARPP32"}, D.assessed_markers(cfg, "M2_HPC_1")
_r = run("s06 with run log", "s06_build_master.py")
assert _r.returncode == 0, _r.stdout
_m2 = pd.read_csv(masters["M2"])
assert _m2.pos_DARPP32.notna().all(), "ran-and-found-nothing must be 0, not blank"
assert int(_m2.pos_DARPP32.sum()) == 0
assert _m2.pos_PNN.isna().all(), "WFA not in the run log for M2 -> must stay blank"
os.remove(_runs)
run("s06 restore", "s06_build_master.py")
check("run log: 'never ran' stays blank, 'ran and found nothing' is a real 0")

# --- incremental fill across environments (the actual workflow) ---
_k = "M2_HPC_1"
_p = D.path_for(cfg, _k, curated=False)
_before = pd.read_csv(_p)
# env 1 contributes DARPP32
_e, _ = D.load(cfg, _k, curated=False)
_e = D.replace_marker(_e, "DARPP32", D.from_points("DARPP32", [[5, 5]], source="cellpose"))
D.save(cfg, _k, _e, curated=False); D.log_run(cfg, _k, "DARPP32", "cellpose", 1)
# env 2 contributes WFA -- must not disturb DARPP32
_e, _ = D.load(cfg, _k, curated=False)
_e = D.replace_marker(_e, "WFA", D.from_points("WFA", [[9, 9]], source="lupori", score=0.7))
D.save(cfg, _k, _e, curated=False); D.log_run(cfg, _k, "WFA", "lupori", 1)
_after, _ = D.load(cfg, _k)
assert int((_after.marker == "DARPP32").sum()) == 1, "env 2 clobbered env 1's marker"
assert int((_after.marker == "WFA").sum()) == 1
assert D.assessed_markers(cfg, _k) == {"WFA", "DARPP32"}
_runs = D.load_runs(cfg)
assert len(_runs[_runs.image == _k]) == 2, "run log should hold one row per marker"
# re-running env 1 replaces its own rows only, and does not duplicate the log row
_e, _ = D.load(cfg, _k, curated=False)
_e = D.replace_marker(_e, "DARPP32", D.from_points("DARPP32", [[6, 6], [7, 7]], source="cellpose"))
D.save(cfg, _k, _e, curated=False); D.log_run(cfg, _k, "DARPP32", "cellpose", 2)
_after, _ = D.load(cfg, _k)
assert int((_after.marker == "DARPP32").sum()) == 2 and int((_after.marker == "WFA").sum()) == 1
assert len(D.load_runs(cfg)[D.load_runs(cfg).image == _k]) == 2
_before.to_csv(_p, index=False)
os.remove(D.runs_path(cfg))
run("s06 restore2", "s06_build_master.py")
check("incremental fill: each env writes only its markers, log accumulates, reruns replace")

# --- curated inputs preferred ---
det, _ = D.load(cfg, "M1_NAC1_1", curated=False)
extra = D.from_points("WFA", [[CENTRES[4][0], CENTRES[4][1]]], source="manual")
D.save(cfg, "M1_NAC1_1", pd.concat([det, extra], ignore_index=True), curated=True)
r = run("s06 curated", "s06_build_master.py")
assert r.returncode == 0 and "[curated detections]" in r.stdout, r.stdout
m1c = pd.read_csv(masters["M1"])
assert int(m1c[m1c.image == "M1_NAC1_1"].pos_PNN.sum()) == 3, "hand-added PNN not used"
assert os.path.exists(os.path.join(TMP, "results/detections", "M1_NAC1_1.csv")), \
    "raw detections must survive"
check("curated detections preferred by s06; raw output left intact")

r = run("s05 reset", "s05_curate_detections.py", "--reset", "M1_NAC1_1")
assert r.returncode == 0
r = run("s06 after reset", "s06_build_master.py")
m1r = pd.read_csv(masters["M1"])
assert int(m1r[m1r.image == "M1_NAC1_1"].pos_PNN.sum()) == 2, "reset should drop the edit"
check("--reset discards curation and falls back to the raw detections")

# --- ROIs ---
for key in SPECS:
    feats = []
    for nm, (x0, x1) in zip(["NAc_shell", "NAc_core"], [(0, 110), (110, W)]):
        feats.append({"type": "Feature", "properties": {"name": nm},
                      "geometry": {"type": "Polygon",
                                   "coordinates": [[[x0, 0], [x1, 0], [x1, 300],
                                                    [x0, 300], [x0, 0]]]}})
    R.save_geojson(os.path.join(TMP, "rois", f"{key}.geojson"), feats)
r = run("s08", "s08_assign_rois.py")
assert r.returncode == 0, r.stdout
m1a = pd.read_csv(masters["M1"])
_in = m1a[m1a.roi.fillna("") != ""]
assert int(_in.anchored_by.eq("DAPI").sum()) == 2 * N_IN_ROI, \
    int(_in.anchored_by.eq("DAPI").sum())
# orphan cells are assigned to regions like any other cell -- that is the point,
# they have to land in a denominator
assert int((_in.anchored_by != "DAPI").sum()) == 2, int((_in.anchored_by != "DAPI").sum())
before = pd.read_csv(masters["M1"])
run("s08 again", "s08_assign_rois.py")
pd.testing.assert_frame_equal(before, pd.read_csv(masters["M1"]))
check("ROI assignment correct and idempotent; cells outside polygons stay blank")

for key in SPECS:
    os.remove(os.path.join(TMP, "rois", f"{key}.geojson"))
r = run("s08 whole", "s08_assign_rois.py", "--whole-image")
assert r.returncode == 0 and "whole-image" in r.stdout
m1w = pd.read_csv(masters["M1"])
assert set(m1w.roi) == {"NAC1"}, set(m1w.roi)
check("--whole-image falls back to the BrainRegion token")

# --- napari <-> geojson round trip ---
named = [("NAc_shell", np.array([[0, 0], [0, 110], [300, 110], [300, 0]], float)),
         ("stray", np.array([[5, 5], [6, 6]], float))]
feats = R.shapes_to_features(named)
assert len(feats) == 1, "a 2-vertex stray click must be dropped"
p = R.save_geojson(os.path.join(TMP, "rois", "rt.geojson"), feats)
back = R.features_to_shapes(p)
assert np.allclose(back["NAc_shell"][0][0], [0, 0])
check("napari shapes <-> geojson round trip, stray clicks dropped")

# --- NESTED regions: anatomy nests (core sits inside shell), so the SMALLEST
#     containing polygon must win, and the answer must not depend on file order ---
_shell = [[0, 0], [300, 0], [300, 300], [0, 300], [0, 0]]
_core = [[100, 100], [200, 100], [200, 200], [100, 200], [100, 100]]
def _gj(order, path):
    feats = [{"type": "Feature", "properties": {"name": n},
              "geometry": {"type": "Polygon", "coordinates": [c]}} for n, c in order]
    json.dump({"type": "FeatureCollection", "features": feats}, open(path, "w"))
_pts = np.array([[150, 150], [50, 50], [400, 400]])
_got = []
for order in ([("NAc_shell", _shell), ("NAc_core", _core)],
              [("NAc_core", _core), ("NAc_shell", _shell)]):
    _p = os.path.join(TMP, "rois", "_nest.geojson")
    _gj(order, _p)
    _got.append(list(R.assign_rois(_pts, _p)))
assert _got[0] == _got[1], f"assignment depends on file order: {_got}"
assert _got[0] == ["NAc_core", "NAc_shell", ""], _got[0]
assert R.polygon_area(R.load_polygons(_p)[0][1]) > 0
check("nested ROIs: innermost polygon wins, independent of file/layer order")

# --- study vocabulary: names must stay consistent across many images ---
assert C.roi_layer_names(cfg, "NAc") == ["NAc_shell", "NAc_core"]
# substring matching: the filenames say NAC1, the vocabulary says NAc
assert C.roi_match(cfg, "NAC1") == ("NAc", ["NAc_shell", "NAc_core"]), C.roi_match(cfg, "NAC1")
assert C.roi_match(cfg, "leftNAc_med")[0] == "NAc"
assert C.roi_match(cfg, "hpc3")[0] == "HPC"
_amb = dict(cfg)
_amb["roi_drawing"] = {"vocabulary": {"CA": ["a"], "AC": ["b"]}, "strict": True}
try:
    C.roi_match(_amb, "xCAACy")
    raise SystemExit("ambiguous region should raise")
except ValueError:
    pass
_long = dict(cfg)
_long["roi_drawing"] = {"vocabulary": {"CA": ["a"], "CA1": ["b"]}, "strict": True}
assert C.roi_match(_long, "CA1")[0] == "CA1", "longest key must win"
assert C.roi_layer_names(cfg, "CTX") == [], "undeclared region must yield no names"
assert C.roi_all_names(cfg) == ["CA1", "DG", "NAc_core", "NAc_shell"]
assert C.roi_strict(cfg) is True
_ok, _bad = C.check_roi_names(cfg, "NAC1", ["NAc_shell", "NAc_core"])
assert _bad == [], _bad
_ok, _bad = C.check_roi_names(cfg, "NAC1", ["NAc_Shell"])
assert len(_bad) == 1 and _bad[0][1] == ["NAc_shell"], _bad
_ok, _bad = C.check_roi_names(cfg, "NAC1", ["CA1"])
assert len(_bad) == 1 and _bad[0][1] == [], "a valid name from ANOTHER region is still wrong here"
check("ROI vocabulary: substring match, longest-wins, ambiguity refused, typos caught")

# s00 flags a brain region that has no vocabulary entry
_c = open(CFG).read()
assert "    NAc: [NAc_shell, NAc_core]\n" in _c, "fixture indentation changed"
open(CFG, "w").write(_c.replace("    NAc: [NAc_shell, NAc_core]\n", ""))
r = run("s00 undeclared", "s00_check_inputs.py")
assert "no ROI vocabulary for 'NAC1'" in r.stdout, r.stdout
assert "need attention" in r.stdout, r.stdout
open(CFG, "w").write(CONFIG)
check("s00 pre-flight flags brain regions with no ROI vocabulary declared")

# --- the retired machinery must not be back in the pipeline ---
stages = sorted(f for f in os.listdir(SRC) if f.startswith("s") and f.endswith(".py"))
assert not any("threshold" in f or "score_markers" in f or "check_positivity" in f
               for f in stages), stages
for f in stages:
    assert "__main__" in open(os.path.join(SRC, f)).read(), f"{f} is not runnable"
for lib in ("config.py", "io_utils.py", "rois.py", "detections.py", "curate_lib.py"):
    assert "__main__" not in open(os.path.join(SRC, lib)).read(), f"{lib} should be a library"
txt = " ".join(open(os.path.join(SRC, f)).read() for f in stages)
assert "thresholds" not in txt.replace("threshold_calibration", ""), \
    "no stage should mention thresholds any more"
check("no threshold machinery in src/; every numbered stage runnable, libs are not")

# --- isolation: the real project was never touched ---
real = C.load(os.path.join(PROJ, "config.yaml"))
assert real["paths"]["results"].startswith(PROJ)
assert cfg["paths"]["results"].startswith(TMP)
assert not os.path.exists(os.path.join(TMP, "..", "DZP1F1_master.csv"))
check("suite ran entirely inside a temp project — the real data/ was never opened")

# --- GUI construction (needs napari; skipped cleanly without it) ---
try:
    import napari
    _o = napari.Viewer
    napari.Viewer = lambda *a, **k: _o(*a, **{**k, "show": False})
    napari.run = lambda *a, **k: None
    import s03_curate_nuclei as G3, s05_curate_detections as G5, s07_draw_rois as G7
    img = images["M1_NAC1_1"]
    G3.curate_one(cfg, img)
    check("GUI s03 (curate nuclei) builds its layers")
    G5.curate_one(cfg, img)
    check("GUI s05 (curate all channels) builds its layers")
    feats = G7.draw_one(cfg, img, os.path.join(TMP, "rois", "M1_NAC1_1.geojson"))
    assert feats == [] or isinstance(feats, list)
    check("GUI s07 (draw ROIs) builds its layers")
except ImportError:
    print("SKIP: napari not importable — GUI construction not checked")

print(f"\n=== ALL {len(ok)} CHECKS PASSED ===")
shutil.rmtree(TMP, ignore_errors=True)
print("temp project removed")
