"""
Stage 2 — DAPI nuclei = the anchor cell set.
Segment nuclei from each image's DAPI channel file with Cellpose ('nuclei' model).
Saves one integer label mask per IMAGE (not per file) to results/nuclei/<key>.npy,
where <key> is MouseID_BrainRegion_Slice#.

Run in the `seg` env (Cellpose/StarDist). Swap in StarDist if you prefer.
    python src/s02_detect_nuclei.py                    # every image in data/
    python src/s02_detect_nuclei.py data/M1_NAc_1_1.tif  # just that image
"""
import os
import sys
import numpy as np
import config as C
import io_utils as io

NUC_DIAM_PX = None      # None = Cellpose auto-estimate; set an int for speed/consistency
GPU = None              # None = use MPS/CUDA if torch has one (much faster); True/False to force
MODEL_V3 = "nuclei"     # cellpose <=3: the dedicated nuclei model
MODEL_V4 = "cpsam"      # cellpose >=4: the generalist model (v4 dropped 'nuclei')

_cache = {}


def _model():
    """Build the Cellpose model, tolerating the v3 -> v4 API break.

    v4 renamed Cellpose -> CellposeModel, dropped model_type='nuclei' in favour of one
    generalist model, and dropped the `channels` argument. Cached so a multi-image run
    loads weights once.
    """
    if "m" in _cache:
        return _cache["m"]
    import cellpose
    from cellpose import models
    ver = str(getattr(cellpose, "version", "?"))

    gpu = GPU
    if gpu is None:     # CPU is ~20x slower here, so take an accelerator if there is one
        try:
            import torch
            gpu = bool(torch.cuda.is_available()
                       or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:
            gpu = False

    if hasattr(models, "Cellpose"):                     # v3 and earlier
        model, kind = models.Cellpose(model_type=MODEL_V3, gpu=gpu), f"v{ver} '{MODEL_V3}'"
        kwargs = {"channels": [0, 0]}
    else:                                               # v4+
        model = models.CellposeModel(gpu=gpu, pretrained_model=MODEL_V4)
        kind, kwargs = f"v{ver} '{MODEL_V4}'", {}
    print(f"  cellpose {kind}, gpu={gpu}")
    _cache["m"] = (model, kwargs)
    return _cache["m"]


def segment(dapi):
    model, kwargs = _model()
    # v3 returns (masks, flows, styles, diams); v4 returns (masks, flows, styles)
    out = model.eval(dapi, diameter=NUC_DIAM_PX, **kwargs)
    masks = out[0] if isinstance(out, tuple) else out
    return np.asarray(masks).astype(np.int32)


def main(paths=None):
    cfg = C.load()
    images = C.discover(cfg, paths)
    if not images:
        sys.exit(f"no images found in {cfg['paths']['data']} — expected TIFFs named "
                 f"MouseID_BrainRegion_Slice#_Channel#.tif")
    outdir = os.path.join(cfg["paths"]["results"], "nuclei")
    os.makedirs(outdir, exist_ok=True)

    for key, img in images.items():
        if not img.has("DAPI"):
            print(f"skip {key}: no DAPI channel (present: {img.roles})")
            continue
        dapi = io.channel(cfg, img, "DAPI")
        masks = segment(dapi)
        np.save(os.path.join(outdir, f"{key}.npy"), masks)
        print(f"{key}: {masks.max()} nuclei -> {outdir}/{key}.npy")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
