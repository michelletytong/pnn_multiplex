# The `cpn` environment — the Ciampi/Lupori pretrained PNN detector

Build it from `environment/cpn.yml`:

```bash
conda env create -f environment/cpn.yml
conda activate cpn
```

Two things are **not** in the env file because they are downloads, not packages.

## 1. Their repository

`s04 --detector lupori` shells out to `predict.py` in
[ciampluca/counting_perineuronal_nets](https://github.com/ciampluca/counting_perineuronal_nets).
Clone it anywhere, then point `config.yaml` at it:

```bash
git clone https://github.com/ciampluca/counting_perineuronal_nets
```

```yaml
# config.yaml
paths:
  cpn_repo: /path/to/counting_perineuronal_nets   # <- absolute path, machine-specific
```

## 2. The pretrained weights (~330 MB)

From their [releases](https://github.com/ciampluca/counting_perineuronal_nets/releases),
unzipped into `model_zoo/`:

| version | detection weights | check out repo tag |
|---|---|---|
| v0.5 *(used here)* | `pnn_v2_fasterrcnn_640.zip` | `v0.5` |
| v0.3 *(published atlas)* | `pnn_fasterrcnn_640.zip` | `v0.3` |

⚠️ **The repo version must match the weights.** Point `config.yaml`
`paths.pnn_model` at the unzipped folder, e.g. `model_zoo/pnn_v2_fasterrcnn_640`.

## Notes on the environment

**Why python 3.8.** Their code was written against torch 1.11, and torch 1.11 has no
wheel for newer python.

**torchsort and spacecutter are deliberately omitted**, despite being in their
`requirements.txt`. They are used only by the *scoring* model (`methods/rank`); this
pipeline runs detection only. Both compile from source, which needs Xcode command
line tools on macOS or Visual C++ Build Tools on Windows — leaving them out removes
that requirement. If you ever want `predict.py --rescore`, add `torchsort==0.1.9` and
`spacecutter==0.2.1` and expect a compile step.

**scikit-learn IS required**, even for detection only: their `datasets/__init__.py`
imports the Rank dataset, which imports sklearn, so it loads no matter which method
you run. This is not obvious and cost a build to discover.

## Verified

Built from `cpn.yml` on macOS arm64 (2026-09-20) and run against
`pnn_v2_fasterrcnn_640`. Produced coordinates identical to the hand-built
environment. Not yet tested on Windows or Linux.
