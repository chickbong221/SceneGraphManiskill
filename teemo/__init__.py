"""TEEMO spatio-temporal semantic graph package.

Sub-modules
-----------
vocab               : SINGLE SOURCE OF TRUTH for label sets, eligible pairs, counts.
graph_builder       : build_graph(...), extract_state(...), extract_z_features(...),
                      TeemoGraphSpec, GRAPH_DIM, PAIRS, SLOT_ORDER.
history             : GraphHistory ring buffer (auto-reset-safe).
affordance_use      : AffordanceBank — runtime nearest-candidate alignment/aperture.
affordance_extract  : OFFLINE; success demos -> per-class object-frame candidates.
calibrate_thresholds: OFFLINE; demos -> thresholds.json.
encoders            : RelationGraphEncoder, CausalRelationMask, GraphAuxiliaryHeads.
viz_graph           : OFFLINE; RGB overlay + node-link figures.

Runtime dependencies
--------------------
torch, numpy, gymnasium, mani_skill  (already in repo)
transforms3d                          (ManiSkill dep, already installed)

Optional / offline-only
-----------------------
h5py            (affordance_extract, calibrate_thresholds)
scikit-learn    (affordance_extract k-means reduction; falls back to subsample)
networkx        (viz_graph node-link diagrams)
matplotlib      (viz_graph, calibration histograms)
opencv-python   (viz_graph RGB overlay)

Install for offline pipeline:
    pip install h5py networkx scikit-learn matplotlib opencv-python
"""

from . import vocab  # noqa: F401
