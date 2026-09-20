# pnn_multiplex pipeline. Envs: seg (Cellpose), cpn (Lupori/torch), analysis.
#
# Input: one TIFF per channel in data/, named MouseID_BrainRegion_Slice#_Channel#.tif
# The number in each src/ filename is the run order.
#
# DAPI anchors everything: it is segmented (s02) and curated (s03) FIRST, because a
# nucleus missing there is a cell missing from the whole study. Then every other
# marker is detected as objects (s04) and curated together in one window (s05).
# Every pos_ column comes from a detected object matched 1:1 to a nucleus -- there
# are no intensity cutoffs in the pipeline.
#
#     make check          s00
#     make nuclei         s02   [seg env]
#     make curate-nuclei  s03   GUI: fix the anchor
#     make detect-markers s04   cellpose channels, then lupori in the cpn env
#     make curate         s05   GUI: all 4 channels, editable
#     make master         s06
#     make rois           s07   GUI: draw regions
#     make assign         s08   -> results/<MouseID>_master.csv, the end product

.DEFAULT_GOAL := help
.PHONY: help check samples nuclei curate-nuclei detect-markers detect-cellpose \
        detect-lupori detect-status curate master rois assign assign-whole clean clean-curation

help:        ## show this list
	@echo "pnn_multiplex — stages run in the order numbered in src/"
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

check:          ## s00: what the pipeline sees in data/ — grouping + channel roles
	python src/s00_check_inputs.py

samples:        ## s01 optional: starter samples.csv, one row per mouse
	python src/s01_make_samples.py

nuclei:         ## s02: DAPI -> nuclei masks              [seg env]
	python src/s02_detect_nuclei.py

curate-nuclei:  ## s03 GUI: curate the DAPI nuclei — the anchor. Do this before s04
	python src/s03_curate_nuclei.py

detect-cellpose: ## s04a: cellpose-detected markers (DARPP32, ERa)   [analysis/seg env]
	python src/s04_detect_markers.py --detector cellpose

detect-lupori:  ## s04b: the PNN detector (WFA)                      [cpn env]
	python src/s04_detect_markers.py --detector lupori

detect-markers: ## s04: run whatever detectors THIS env supports, report the rest
	python src/s04_detect_markers.py

detect-status:  ## s04: what is detected and what is still outstanding (runs nothing)
	python src/s04_detect_markers.py --status

curate:         ## s05 GUI: check/fix ALL 4 channels' detections, editable
	python src/s05_curate_detections.py

master:         ## s06: per-mouse master table (uses curated nuclei + detections)
	python src/s06_build_master.py

rois:           ## s07 GUI: draw your brain regions in napari
	python src/s07_draw_rois.py

assign:         ## s08: stamp drawn ROIs onto cells; re-run whenever you redraw
	python src/s08_assign_rois.py

assign-whole:   ## s08 without s07: treat each whole crop as one region
	python src/s08_assign_rois.py --whole-image

clean:          ## delete detections + tables (KEEPS curation and ROIs)
	rm -rf results/nuclei results/detections results/*_master.csv

clean-curation: ## discard ALL manual curation (keeps raw detections)
	python src/s03_curate_nuclei.py --reset
	python src/s05_curate_detections.py --reset
