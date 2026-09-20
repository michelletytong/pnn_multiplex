"""Load config.yaml, and turn a directory of per-channel TIFFs into IMAGES.

Input layout: one TIFF per channel, named MouseID_BrainRegion_Slice#_Channel#.tif.
The files sharing MouseID_BrainRegion_Slice# are one image; discover() groups them
and maps each Channel# to a marker role via the global `channels` map.
"""
import os
import re
import glob
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path=None):
    """Load config.yaml.

    $PNN_CONFIG overrides the default location, so a test run can point the whole
    pipeline at a throwaway project directory and never touch your real data/,
    results/ or rois/. (A test that operated on the live directories is how a set of
    hand-drawn ROI polygons got deleted once.)
    """
    path = path or os.environ.get("PNN_CONFIG") or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # relative paths resolve against the CONFIG FILE's directory, not the package —
    # so a config placed in a temp project directory keeps everything inside it
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = v if os.path.isabs(v) else os.path.join(base, v)
    # channel keys may come back as str depending on how they were written
    cfg["channels"] = {int(k): v for k, v in cfg["channels"].items()}
    cfg["_root"] = base
    return cfg


# --------------------------------------------------------------------------
# filename parsing / image discovery
# --------------------------------------------------------------------------

def parse_name(cfg, stem):
    """'M1_NAc_1_2' -> {animal, region, slice, channel, key}, or None if it doesn't match."""
    m = re.match(cfg["filename_pattern"], stem)
    if not m:
        return None
    d = m.groupdict()
    d["slice"] = int(d["slice"])
    d["channel"] = int(d["channel"])
    d["key"] = f"{d['animal']}_{d['region']}_{d['slice']}"
    return d


class Image(dict):
    """One image = the channel files sharing MouseID_BrainRegion_Slice#.

    Keys: key, animal, region, slice, files {role: path}.
    """

    def path(self, role):
        return self["files"].get(role)

    def has(self, role):
        return role in self["files"]

    @property
    def roles(self):
        return sorted(self["files"])


def discover(cfg, paths=None, quiet=False):
    """Group per-channel TIFFs into images. -> dict key -> Image, in sorted key order."""
    paths = paths or sorted(glob.glob(os.path.join(cfg["paths"]["data"], "*.tif*")))
    chmap = cfg["channels"]
    images, skipped, unknown_ch = {}, [], set()

    for p in sorted(paths):
        stem = os.path.splitext(os.path.basename(p))[0]
        d = parse_name(cfg, stem)
        if d is None:
            skipped.append(os.path.basename(p))
            continue
        role = chmap.get(d["channel"])
        if role is None:
            unknown_ch.add(d["channel"])
            continue
        img = images.setdefault(d["key"], Image(
            key=d["key"], animal=d["animal"], region=d["region"],
            slice=d["slice"], files={}))
        if role in img["files"]:
            raise ValueError(
                f"two files map to role '{role}' for image '{d['key']}':\n"
                f"  {img['files'][role]}\n  {p}")
        img["files"][role] = p

    if not quiet:
        if skipped:
            print(f"  [!] {len(skipped)} file(s) did not match filename_pattern and were "
                  f"ignored: {skipped[:4]}{' ...' if len(skipped) > 4 else ''}")
        if unknown_ch:
            print(f"  [!] channel number(s) {sorted(unknown_ch)} are not in config.yaml "
                  f"`channels` ({dict(chmap)}) — those files were ignored.")
    return {k: images[k] for k in sorted(images)}


def require(img, role):
    """Path for a role, or a clear error naming the file that should exist."""
    p = img.path(role)
    if p is None:
        raise FileNotFoundError(
            f"image '{img['key']}' has no {role} channel "
            f"(present: {img.roles or 'none'}). Expected a file "
            f"'{img['key']}_<Channel#>.tif' whose Channel# maps to {role}.")
    return p


# --------------------------------------------------------------------------
# per-animal metadata (samples.csv)
# --------------------------------------------------------------------------

_MANIFEST = None   # module cache: DataFrame indexed by animal, or False if none


def _manifest(cfg):
    global _MANIFEST
    if _MANIFEST is None:
        import pandas as pd
        p = cfg["paths"].get("manifest")
        if p and os.path.exists(p):
            m = pd.read_csv(p, dtype=str)
            if "animal" not in m.columns:
                raise ValueError(f"{p} must have an 'animal' column (one row per mouse); "
                                 f"found {list(m.columns)}")
            _MANIFEST = m.set_index("animal")
        else:
            _MANIFEST = False
    return _MANIFEST if _MANIFEST is not False else None


def meta_of(cfg, img):
    """Per-image metadata: animal/brain_region/slice from the filename, plus any
    per-animal columns (condition, sex, ...) from samples.csv. Every cell in s06 is
    stamped with this, so cells are always traceable to an animal.

    Note brain_region here is the coarse region baked into the filename. The fine
    region a cell sits in is the `roi` column, assigned from your polygons in s08.
    """
    import pandas as pd
    meta = {"animal": img["animal"], "brain_region": img["region"], "slice": img["slice"]}
    m = _manifest(cfg)
    if m is not None and img["animal"] in m.index:
        row = m.loc[img["animal"]]
        if isinstance(row, pd.DataFrame):        # duplicate animal rows in the sheet
            raise ValueError(f"samples.csv has {len(row)} rows for animal "
                             f"'{img['animal']}' — it should have exactly one.")
        meta.update({k: v for k, v in row.items() if pd.notna(v)})
    return meta


INTENSITY_COMPARTMENTS = ("nucleus", "soma_ring")


def markers(cfg):
    """The non-anchor markers, in config order. DAPI is the anchor and is not one."""
    return dict(cfg.get("markers") or {})


def detector_of(cfg, marker):
    """'lupori' | 'cellpose' — or None for a marker that is measured, not detected."""
    return (markers(cfg).get(marker) or {}).get("detect")


def detected_markers(cfg):
    """Markers that produce objects, and therefore a pos_ column. A marker with no
    `detect` is measured only: reporting its graded intensity says more than a bit."""
    return [m for m in markers(cfg) if detector_of(cfg, m)]


def measure_in(cfg, marker):
    """The compartments this marker is measured in, always a list."""
    v = (markers(cfg).get(marker) or {}).get("measure")
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def fluo_col(cfg, marker, compartment):
    """Column name for one measurement. A marker measured in several compartments
    gets one suffixed column each; a single compartment keeps the bare name."""
    comps = measure_in(cfg, marker)
    return f"fluoMean_{marker}" if len(comps) <= 1 else f"fluoMean_{marker}_{compartment}"


def fluo_cols(cfg, marker):
    return [fluo_col(cfg, marker, c) for c in measure_in(cfg, marker)]


def union_anchor(cfg):
    return bool((cfg.get("detection") or {}).get("union_anchor", False))


def call_name(cfg, marker):
    """Name of the marker's 0/1 column, without the pos_ prefix. A WFA detection is
    a perineuronal net, so it reads pos_PNN rather than pos_WFA."""
    return (markers(cfg).get(marker) or {}).get("call_as") or marker


def intensity_markers(cfg):
    """Markers with an intensity compartment -> the stable fluoMean_ columns."""
    return [m for m in markers(cfg)
            if any(c in INTENSITY_COMPARTMENTS for c in measure_in(cfg, m))]


def roi_vocabulary(cfg):
    """{BrainRegion: [allowed ROI names]} — the study's declared regions."""
    return dict((cfg.get("roi_drawing") or {}).get("vocabulary") or {})


def roi_match(cfg, region):
    """Which vocabulary entry covers this BrainRegion token. -> (key, names).

    Matched by SUBSTRING, case-insensitively, so a token of 'NAC1', 'NAc_med' or
    'leftNAc' all resolve to the 'NAc' entry — you declare a region once, however
    your filenames happen to spell it.

    Resolution order, most specific first:
      1. exact token match
      2. case-insensitive exact
      3. longest vocabulary key contained in the token
    Two keys matching at the same length is ambiguous and raises, rather than
    silently picking one.
    """
    vocab = roi_vocabulary(cfg)
    if region in vocab:
        return region, list(vocab[region])
    low = (region or "").lower()
    for k in vocab:
        if k.lower() == low:
            return k, list(vocab[k])
    hits = [k for k in vocab if k.lower() in low]
    if not hits:
        return None, []
    longest = max(len(k) for k in hits)
    best = [k for k in hits if len(k) == longest]
    if len(best) > 1:
        raise ValueError(
            f"BrainRegion '{region}' matches several vocabulary entries {sorted(best)} "
            f"equally well. Rename one, or add an exact entry for '{region}'.")
    return best[0], list(vocab[best[0]])


def roi_layer_names(cfg, region):
    """ROI names declared for a brain region, or [] if nothing in the vocabulary
    covers it."""
    return roi_match(cfg, region)[1]


def roi_all_names(cfg):
    """Every ROI name declared anywhere — used to spot cross-region typos."""
    out = []
    for names in roi_vocabulary(cfg).values():
        out.extend(names)
    return sorted(set(out))


def roi_strict(cfg):
    return bool((cfg.get("roi_drawing") or {}).get("strict", True))


def check_roi_names(cfg, region, names):
    """Validate drawn ROI names against the vocabulary.
    -> (ok_names, [(bad_name, [close matches])]). Empty second list means all fine."""
    import difflib
    allowed = roi_layer_names(cfg, region)
    pool = allowed or roi_all_names(cfg)
    bad = []
    for n in names:
        if allowed and n in allowed:
            continue
        if not allowed and n in pool:
            continue
        bad.append((n, difflib.get_close_matches(n, pool, n=3, cutoff=0.6)))
    return [n for n in names if n not in [b for b, _ in bad]], bad
