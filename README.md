# pnn_multiplex

Counts cells in multiplexed fluorescence images and tells you, **for every single
cell**, which markers it has and where in the brain it sits.

**You get one spreadsheet per mouse. One row per cell. A column for each marker.**

It leans on two excellent tools that other people wrote — Cellpose for cell counting
or soma/nuclear labels, and the Ciampi/Lupori PNN detector for perineuronal nets.
**Please cite them** if you publish anything using this; links and papers are under
[Tools that we relied on](#tools-that-we-relied-on).

MIT licensed — use it however you like.

---

## The one idea you need

Everything is anchored on **DAPI nuclei**. The pipeline finds every nucleus, and that
becomes the list of cells. Then for each marker it either:

- **measures** how bright that marker is around the cell, or
- **detects** objects (a perineuronal net, a DARPP-32⁺ soma) and **matches** each to
  a cell — or both.

So "does this cell have a PNN?" is answered by *finding a net and matching it to a
nucleus*, not by asking whether the WFA channel is bright there. That matters: WFA
stains diffusely across the tissue, so brightness alone would call half the field
positive.

---

## Platform support

| | status |
|---|---|
| **macOS (Apple Silicon)** | tested — the pipeline and all three environments were built and run here |
| **macOS (Intel), Linux** | should work; unverified |
| **Windows** | Python code is platform-independent, but `make` is not installed by default — see [Windows](#windows). Environments unverified. |

If you get it running somewhere new, please open an issue saying so — including
anything you had to change. `python tests/suite.py` is a quick way to check the
pipeline logic works on your machine.

## Installing

You need [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
(Miniconda is fine). Then:

```bash
git clone <this repo>
cd pnn_multiplex
conda env create -f environment/analysis.yml
conda activate pnn-analysis
```

That's enough to run most of the pipeline. **The two detectors need their own
environments**, because their dependencies conflict with each other:

```bash
conda env create -f environment/seg.yml     # Cellpose: nuclei and DARPP-32
conda env create -f environment/cpn.yml     # the PNN detector
```

| you want to… | environment |
|---|---|
| everything except detection (including all the windows you click in) | `pnn-analysis` |
| find nuclei, find DARPP-32 | `pnn-seg` |
| find PNNs | `cpn` |

**If you don't care about PNNs, skip the `cpn` environment entirely** — nothing else
needs it.

### One extra step for PNNs

The PNN detector is someone else's code and model, so you fetch them separately:

```bash
# 1. their repository, next to this one (the default config expects it there)
cd ..
git clone https://github.com/ciampluca/counting_perineuronal_nets
cd pnn_multiplex
```

2. their pretrained weights — download
[`pnn_v2_fasterrcnn_640.zip`](https://github.com/ciampluca/counting_perineuronal_nets/releases/download/v0.5/pnn_v2_fasterrcnn_640.zip)
(~330 MB) and unzip it into `model_zoo/`.

Full detail, including which repo tag matches which weights, is in
[`environment/cpn.md`](environment/cpn.md).

### Check it worked

```bash
conda activate pnn-analysis
make check
```

---

## Setting up your own images

### Name the files correctly

**One TIFF per channel**, named:

```
MouseID _ BrainRegion _ Slice# _ Channel#
```

```
data/
  M1_NAc_1_1.tif      <- mouse M1, NAc, slice 1, channel 1 (DAPI)
  M1_NAc_1_2.tif      <- same image, channel 2 (WFA)
  M1_NAc_1_3.tif      <- channel 3 (DARPP-32)
  M1_NAc_1_4.tif      <- channel 4 (ERα)
```

Files sharing `M1_NAc_1` are treated as **one image**. Slice and channel must be
numbers; the MouseID may contain underscores. If a channel is missing for an image,
just don't create the file — that marker stays blank for those cells rather than
becoming wrong.

### Tell it about your markers

`config.yaml`, two blocks. Which channel is which:

```yaml
channels:
  1: DAPI
  2: WFA
  3: DARPP32
  4: ERa
```

…and what to do with each marker:

```yaml
markers:
  WFA:     {detect: lupori,   measure: soma_ring, call_as: PNN}
  DARPP32: {detect: cellpose, measure: soma_ring}
  ERa:     {measure: [nucleus, soma_ring]}
```

- `detect` — how to find objects. Omit it and the marker is only measured, never
  turned into a yes/no.
- `measure` — where to sample brightness: `nucleus`, `soma_ring` (a ring around the
  nucleus, background pixels only), or both.
- `call_as` — rename the yes/no column. A WFA detection is a perineuronal net, so it
  reads `pos_PNN`.

### Declare your brain regions

```yaml
roi_drawing:
  vocabulary:
    NAc: [NAc_shell, NAc_core]
    HPC: [CA1, CA2, CA3, DG]
```

The key is matched as a case-insensitive substring of the BrainRegion part of your
filenames, so `NAC1`, `NAc` and `nac_medial` all find the `NAc` entry. Declaring the
names here stops you typing `NAc_Shell` on one image and `NAc_shell` on the next and
silently ending up with two different regions.

---

## Running it

`make help` lists everything. On **Windows**, `make` isn't installed by default —
see [Windows](#windows) below.

### Step 0 — check your files

```bash
conda activate pnn-analysis
make check
```

Reads nothing but filenames. Tells you how files grouped into images, which marker
each channel became, and what's missing. **Fix problems here**, before spending
twenty minutes on detection.

### Step 1 — record your mice *(optional)*

```bash
make samples
```

Writes `samples.csv`, one row per mouse. Fill in `condition`, `sex`, anything else;
every column you add appears on every cell of that mouse. Skip it and you still get
`animal`, `brain_region` and `slice` from the filenames.

### Step 2 — find the nuclei

```bash
conda activate pnn-seg
make nuclei
```

### Step 3 — check the nuclei ← **a window opens**

```bash
conda activate pnn-analysis
make curate-nuclei
```

The most important manual step: every cell in your final spreadsheet is one of these
nuclei, so one missed here is a cell missing from the whole study.

### Step 4 — find the markers

```bash
conda activate pnn-seg
make detect-cellpose          # DARPP-32 objects

conda activate cpn
make detect-lupori            # PNNs
```

Order doesn't matter, and neither overwrites the other's results. `make detect-status`
shows what's done and what's outstanding.

Two shortcuts you may not need: `make detect-markers` runs whichever detectors the
current environment supports and reports the rest, and `make assign-whole` skips
step 7 by treating each whole image as one region — useful before you've drawn any
outlines.

### Step 5 — check the markers ← **a window opens**

```bash
conda activate pnn-analysis
make curate
```

All channels in one window, everything editable. **Expect real work here** — on our
test image the PNN detector found 23 nets and a human found 14 more.

### Step 6 — build the spreadsheet

```bash
make master
```

### Step 7 — outline your regions ← **a window opens**

```bash
make rois
```

### Step 8 — assign cells to regions

```bash
make assign
```

**Done.** Results are in `results/`.

---

## Working in the windows

Three steps open [napari](https://napari.org). Same conventions throughout:

- **Layers are listed on the left.** The eye icon shows/hides. Clicking a layer's
  *name* selects it — and you edit whichever layer is selected. This is the single
  most common thing to get wrong.
- **Close the window to save.** There is no save button; closing *is* the save.
- Nothing overwrites the detector's output. Your edits go to separate `*_curated`
  files, and `--reset` puts things back.

### Adding and removing nuclei (steps 3 and 5)

| layer | what to do |
|---|---|
| `nuclei` | what Cellpose found. Paint or erase for small corrections. |
| `nuclei: add` | select it, then **click on a cell the computer missed** |
| `nuclei: remove` | select it, then **click on something that isn't a cell** |

The click layers are easier than painting: a click in `nuclei: add` drops in a
nucleus the typical size of the others, and one in `nuclei: remove` deletes the whole
nucleus under your cursor.

### Adding and removing marker objects (step 5)

Each detected marker has its own layer of points — `WFA objects`, `DARPP32 objects`.

1. Click that marker's layer to select it.
2. Turn on the matching image layer so you can see what you're judging.
3. **Add:** the add-point tool (`+` in the toolbar), then click.
4. **Remove:** the select tool, click the point, press <kbd>Delete</kbd>.

Markers that are only *measured* (no `detect` in config) have no object layer.

### Drawing regions (step 7)

1. Select the layer for the region you want.
2. Polygon tool, click around the region, <kbd>Esc</kbd> to close it.
3. Next region in its own layer.

**Draw each region at full size, even where they overlap.** NAc core sits inside NAc
shell — so outline the whole shell, then outline core inside it. Cells in the overlap
go to the *smaller* region automatically. Don't draw shell as a ring with a hole.

The **layer name becomes the region name**; double-click a layer to rename it.

---

## Your results

`results/<MouseID>_master.csv`, one row per cell:

| column | what it is |
|---|---|
| `cellID` | unique name for this cell |
| `animal`, `brain_region`, `slice` | from the filename |
| `condition`, `sex`, … | from `samples.csv`, if you made one |
| `x`, `y` | position in pixels |
| `area_px` | size of the nucleus |
| `roi` | region you drew it into (blank = outside all of them) |
| `fluoMean_<marker>` | brightness. Suffixed per compartment if a marker is measured in more than one (`fluoMean_ERa_nucleus`, `fluoMean_ERa_soma_ring`) |
| `pos_<marker>` | **1** = an object of that marker was matched to this cell |
| `anchored_by` | `DAPI` for normal cells (see below) |
| `<marker>_dist` | distance to the nearest object of that marker |

Plus `results/roi_areas.csv` with each region's area in mm², so density is just
`count / area_mm2`.

### The single most important rule

**A blank is not a zero.**

- `pos_PNN = 0` means *we looked, and there was no net on this cell*.
- `pos_PNN` blank means *we never looked* — that detector wasn't run on that image.

Treat blanks as zeros and you'll report confident negatives for cells nobody
examined. Every spreadsheet tool will happily do this for you. In pandas, `.dropna()`
on that column before counting.

### Cells with no nucleus

Your section is a thin slice through 3D tissue, so sometimes you catch a cell's
cytoplasm while its nucleus sits above or below — a DARPP-32⁺ soma with no DAPI in it.

Those cells are kept, marked `anchored_by = DARPP32` instead of `DAPI`, because
discarding them biases densities (it happens more in dense tissue). But they have
**no nuclear measurements** — there is no nucleus to measure — so nucleus-compartment
columns and `area_px` are blank.

To use only cells with a visible nucleus: filter `anchored_by == "DAPI"`.

This recovers the cells you can see. It does nothing for the reverse case — a nucleus
in plane whose marker is out of it — which reads as a genuine negative. Only z-stacks
fix that.

---

## Things that will trip you up

**Whether a marker is nuclear is worth checking yourself.** In our test images the
ERα signal was only 1.14× brighter inside nuclei than outside, and the ring around
the nucleus was brighter than the nucleus itself in 85% of cells — not what you'd
expect of a nuclear receptor. That's why `config.yaml` can measure a marker in
several compartments at once. Look at your own images before deciding which column
to trust.

**Not every marker deserves a yes/no.** If a marker's brightness is a smooth gradient
with no natural dividing line, any cutoff you pick determines the answer. Leave
`detect` out of its config entry and report the intensity instead.

**Image Pre-processing** If each image was auto-brightened separately on export, brightness
isn't comparable *between* images. Raw 16-bit avoids the question.

---

## If something goes wrong

| problem | fix |
|---|---|
| `make check`: "did not match filename_pattern" | a filename is wrong — see the naming rule |
| `make check`: "no ROI vocabulary for X" | add that brain region to `roi_drawing.vocabulary` |
| "napari is not installed in this env" | `conda activate pnn-analysis` |
| "this env cannot run --detector …" | wrong environment for that detector |
| "PNN model not found" | weights not unzipped into `model_zoo/` — see `environment/cpn.md` |
| A window opened but looks empty | click the eye icons — most layers start hidden |
| `OMP: Error #15` | you installed into a mixed conda/pip env; rebuild from the `.yml` |
| You edited the wrong layer | re-run that step and fix it; nothing is final |
| You want your edits gone | `make clean-curation` |

`make clean` deletes the computer's results but **keeps** everything you did by hand.

### Windows

`make` isn't installed on Windows by default. Either install it
([Chocolatey](https://chocolatey.org/): `choco install make`, plus Git Bash so the
Makefile's `grep`/`awk`/`rm` work), or run the scripts directly — every `make X` maps
to one command:

```
make check           python src/s00_check_inputs.py
make samples         python src/s01_make_samples.py
make nuclei          python src/s02_detect_nuclei.py
make curate-nuclei   python src/s03_curate_nuclei.py
make detect-cellpose python src/s04_detect_markers.py --detector cellpose
make detect-lupori   python src/s04_detect_markers.py --detector lupori
make curate          python src/s05_curate_detections.py
make master          python src/s06_build_master.py
make rois            python src/s07_draw_rois.py
make assign          python src/s08_assign_rois.py
```

The Python itself is platform-independent; see [Platform support](#platform-support)
for what has actually been verified.

---

## Running it again

Nothing needs starting over. Each step redoes only its own work:

- Changed a region outline → `make rois`, then `make assign`.
- Added nuclei → `make master` again (a second or two).
- New mouse → put the files in `data/` and run from step 0; finished images are
  skipped.

---

## Tests

```bash
conda activate pnn-analysis
python tests/suite.py
```

Builds a throwaway project in a temp directory, runs the pipeline against synthetic
images, and checks 26 things. It never touches your `data/`, `results/` or `rois/`.
Worth running after you change anything.

---

## Tools that we relied on

**Cellpose** — segments nuclei (step 2) and DARPP-32 somata (step 4). We use the
`cpsam` model.
[github.com/MouseLand/cellpose](https://github.com/MouseLand/cellpose)

> Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman
> generalization for cellular segmentation. *bioRxiv.*
> [10.1101/2025.04.28.651001](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)
>
> Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a
> generalist algorithm for cellular segmentation. *Nature Methods, 18*(1), 100–106.

**counting_perineuronal_nets** — the pretrained PNN detector (step 4). The model and
the inference code are theirs; we only call it and match its output to nuclei.
[github.com/ciampluca/counting_perineuronal_nets](https://github.com/ciampluca/counting_perineuronal_nets)
(Apache 2.0)

Two papers, covering different parts of it — **please cite both**:

> **The detection method** — Ciampi, L., Santiago, C., Costeira, J. P., Gennaro, C.,
> & Amato, G. (2022). Learning to count biological structures with raters'
> uncertainty. *Medical Image Analysis, 80*, 102500.
> [10.1016/j.media.2022.102500](https://doi.org/10.1016/j.media.2022.102500)
>
> **The atlas that applied it, and the PNN work itself** — Lupori, L., et al.
> (2023). A Comprehensive Atlas of Perineuronal Net Distribution and Colocalization
> with Parvalbumin in the Adult Mouse Brain. *bioRxiv.*
> [10.1101/2023.01.24.525313](https://www.biorxiv.org/content/10.1101/2023.01.24.525313v1)

The nucleus-matching approach here — localise each net, then assign it to the nearest
cell — follows what Lupori et al. did to match PNNs to parvalbumin neurons.

**napari** — every window you click in.
[napari.org](https://napari.org)

> napari contributors (2019). napari: a multi-dimensional image viewer for Python.
> [10.5281/zenodo.3555620](https://doi.org/10.5281/zenodo.3555620)

Also: [scikit-image](https://scikit-image.org),
[SciPy](https://scipy.org), [pandas](https://pandas.pydata.org),
[NumPy](https://numpy.org).

---

## How this was built

This pipeline was written with [Claude Code](https://claude.com/claude-code)
(Anthropic). Scientific decisions were mine — what to measure and in which
compartments, whether a marker deserves a yes/no at all or just fluorescence measurements, how to handle somata whose nucleus is out of the imaging plane, even the logic of the code itself and how the data *.csv should be organized — and the code was written to those
decisions, then checked against real images by a human user at every step.

## Licence

MIT — see [LICENSE](LICENSE). Use it, change it, redistribute it, no permission
needed.

**But please cite Cellpose, and both papers behind the PNN detector** if this contributes
to work you publish. The segmentation and the net detection are the scientific
substance here; this pipeline is just using them in a way that suits our needs. Papers and links are in
[Tools that we relied on](#tools-that-we-relied-on).

## How it fits together

The steps run in the order numbered in `src/`:

```
s00_check_inputs.py       what the pipeline sees in data/
s01_make_samples.py       starter samples.csv
s02_detect_nuclei.py      DAPI -> nuclei                     [pnn-seg]
s03_curate_nuclei.py      window opens: manual add/remove nuclei
s04_detect_markers.py     PNNs [cpn] + other markers [pnn-seg]
s05_curate_detections.py  window opens: fix every channel
s06_build_master.py       builds the full data CSV, one per mouse
s07_draw_rois.py          window opens: draw your regions
s08_assign_rois.py        assign cells to regions, compute areas
```

`config.py`, `io_utils.py`, `rois.py`, `detections.py` and `curate_lib.py` are shared
code, not steps.

### Notes on pipeline features

**Why anchor on DAPI?** It's the only channel labelling *every* cell. Anchor on a
marker and you can only compute fractions of that marker's cells — no denominator for
"what fraction of all cells".

**Why detect PNNs rather than threshold WFA?** WFA stains extracellular matrix
diffusely, so a bright ring doesn't mean a net. The detector was trained on the
organised structure; intensity can't do that. Hence `fluoMean_WFA` and `pos_PNN` as
separate columns answering separate questions. We are also interested in which cells do/do not have a PNN, in addition to the intensity.

**There are two cell counting steps, technically** cellpose does an initial cell count using the DAPI channel which generates the initial cell nuclei mask. However, when the DARPP32 channel is segmented it gave rise to cells with DARPP32 but no DAPI labeling. This could happen because the DAPI label was outside of the z-plane during imaging. In our data set, we did not take z-stacks. So we opted to have the DARPP32 only cells get added to the "cell" mask as well.

**Why is region area rasterised rather than taken from the outline?** NAc Core sits
inside NAc shell, so the shell *outline* encloses both. To make it easier for the user when they are drawing the ROIs, they can just draw a polygon around the core and the shell. To calculate the area of the shell from the polygons, you would need to subtract the area of the polygon core from the area of the polygon shell or you could draw complicated outlines. To avoid both of these process, the code is written in a way such that if a given pixel has both core and shell roi values, it chooses the smallest polygon (core) assuming containment. So the areas in roi_areas.csv can be used directly in any later calculations (e.g for density) 

The `config.yaml` comments and the docstring atop each `src/` file carry the rest.
