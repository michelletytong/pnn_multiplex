"""Single-channel image IO. Library only — the CLI lives in s00_check_inputs.py.

Each TIFF holds ONE channel; channels are addressed by
ROLE (DAPI/WFA/DARPP32/ERa), resolved from the Channel# in the filename via the
global `channels` map in config.yaml — never by raw index."""
import numpy as np
import tifffile
import config as C


def read(path):
    """Read a single-channel image as float32 (H, W)."""
    a = tifffile.imread(path)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel 2D image, got shape {a.shape}. "
                         f"This pipeline takes ONE TIFF PER CHANNEL "
                         f"(MouseID_BrainRegion_Slice#_Channel#.tif).")
    return a.astype(np.float32)


def channel(cfg, img, role):
    """Read one marker channel of an image, as float32 (H, W)."""
    return read(C.require(img, role))


def describe(path):
    """Print shape/dtype/intensity stats to confirm which marker a Channel# holds."""
    a = np.squeeze(tifffile.imread(path))
    print(f"{path}\n  shape={a.shape} dtype={a.dtype}")
    if a.ndim == 2:
        print(f"  min={a.min():.0f} max={a.max():.0f} mean={a.mean():.1f} "
              f"p99={np.percentile(a, 99):.0f}")
    else:
        print(f"  [!] not 2D — expected one channel per file")
