# Fixation Cloud Object Recovery

This repository contains the analysis code for the fixation-cloud object recovery paper.

The paper asks whether fixation clouds preserve recoverable information about the object being inspected. The model treats gaze as a contracted, object-generated trace: fixation covariance can become organized in the object's coordinate frame, and that geometry can be expanded back toward object extent.

The analyses establish three findings. First, in REFLACX, fixation covariance becomes increasingly organized in the annotated object's coordinate frame during inspection, with later fixations showing stronger axis alignment and object-frame coverage than early fixations. Second, when object-associated fixations are known, the object recovery model reconstructs annotated lesion extent better than foveal coverage and convex hull baselines under patient-level cross-validation. Third, recovery remains possible from unrestricted premention fixation buffers and is strongest when the buffer localizes the mentioned abnormality and contains stable fixation geometry.

## Repository Structure

**Experiment scripts**

`alignment_emergence.py` tests whether within-object fixation covariance becomes organized in the annotated object's coordinate frame. It compares acquisition and extent-sampling fixations, computes alignment and object-frame coverage metrics, and evaluates same-size placebo controls.

`object_recovery.py` tests object-associated lesion extent recovery. It compares foveal coverage, convex hull, and the gain-based object recovery model using patient-level cross-validation.

`mention_recovery.py` tests mention-grounded recovery from unrestricted premention fixation buffers. It evaluates the same recovery methods and summarizes performance by target localization and stable fixation geometry.

`refcoco_validation.py` runs the compact RefCOCO-Gaze comparison using target boxes converted to ellipse-like support regions.

`viz.py` generates the manuscript figures for Experiments 1-3.

**Helper modules**

`reflacxloader.py` loads REFLACX studies, fixations, transcripts, image dimensions, and abnormality ellipses.

`refcocoloader.py` loads RefCOCO-Gaze target episodes and target-aligned gaze records.

`local_annotations.py` defines ellipse annotations and mask/crop utilities.

`fixation_builder.py` converts raw fixation records into analysis-ready fixation objects.

`entity_extraction.py` maps report phrases to abnormality mentions and aligns them with transcript timing.

`stats_utils.py` provides paired patient-level statistics and bootstrap confidence intervals.


