"""
TEEMO graph vocabulary — SINGLE SOURCE OF TRUTH.

Label sets copied VERBATIM from the draft Tables 1 (absolute) and 2 (temporal).
Every other module (graph_builder, viz, calibration, aux heads) imports class
counts and names from here. If this file and the paper table ever disagree, fix
this file first.

Design (faithful to draft §3 + Tables 1-2), StackCube-v1 realization:

NODES (draft §3.1): tcp, gripper, object(s). NO goal node in this version.
  node feature x_i = [ type_onehot(3), class_emb(via id), tau (steps since seen),
                       z_i (masked RGB evidence) ]

ABSOLUTE RELATIONS (draft Table 1), per eligible pair:
  spatial.distance      5-way   tcp-object
  spatial.height        5-way   tcp-object
  alignment.tcp_align   5-way   tcp-object        (via affordance candidate set)
  alignment.aperture    5-way   gripper-object    (via affordance candidate set)
  physical.contact      2-way   gripper-object, object-object
  physical.grasp        2-way   gripper-object
  physical.support      2-way   object-object
  physical.containment  2-way   object-object     [DEFER: const no-containment]

TEMPORAL RELATIONS (draft Table 2), horizon K=5, per eligible pair:
  distance_change       7-way   tcp-object
  height_change         7-way   tcp-object
  alignment_change      7-way   tcp-object
  aperture_change       7-way   gripper-object
  contact_transition    4-way   gripper-object, object-object
  grasp_transition      4-way   gripper-object
  support_transition    4-way   object-object
  containment_transition 4-way  object-object     [DEFER]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Node types (draft §3.1). class embedding is keyed separately by semantic id.
# ---------------------------------------------------------------------------
NODE_TYPES = ("tcp", "gripper", "object")          # 3-way type field
NODE_TYPE_ID = {n: i for i, n in enumerate(NODE_TYPES)}

# Semantic-class vocabulary is DERIVED FROM THE TASK at build time, not fixed.
# We seed it with the StackCube classes; new tasks extend the registry.
SEMANTIC_CLASSES = ("none", "cubeA", "cubeB")      # "none" for tcp/gripper
SEMANTIC_CLASS_ID = {c: i for i, c in enumerate(SEMANTIC_CLASSES)}

# ---------------------------------------------------------------------------
# ABSOLUTE label vocabularies (Table 1) — order defines integer class id.
# ---------------------------------------------------------------------------
DISTANCE_LABELS = ("very-near", "near", "medium", "far", "very-far")           # 5
HEIGHT_LABELS = ("far-below", "below", "level", "above", "far-above")          # 5
TCP_ALIGN_LABELS = ("very-misaligned", "misaligned", "partial",
                    "near-aligned", "aligned")                                  # 5
APERTURE_LABELS = ("too-narrow", "slightly-narrow", "fit",
                   "slightly-wide", "too-wide")                                 # 5
CONTACT_LABELS = ("no-contact", "contact")                                     # 2
GRASP_LABELS = ("no-grasp", "grasp")                                           # 2
SUPPORT_LABELS = ("no-support", "support")                                     # 2
CONTAINMENT_LABELS = ("no-containment", "containment")                         # 2

# ---------------------------------------------------------------------------
# TEMPORAL label vocabularies (Table 2).
# ---------------------------------------------------------------------------
# Continuous 7-way: 3 improve speeds, stable, 3 worsen speeds.
DISTANCE_CHANGE_LABELS = (
    "approach-slow", "approach-medium", "approach-fast",
    "stable-distance",
    "recede-slow", "recede-medium", "recede-fast",
)
HEIGHT_CHANGE_LABELS = (
    "move-up-slow", "move-up-medium", "move-up-fast",
    "stable-height",
    "move-down-slow", "move-down-medium", "move-down-fast",
)
ALIGN_CHANGE_LABELS = (
    "improve-alignment-slow", "improve-alignment-medium", "improve-alignment-fast",
    "stable-alignment",
    "worsen-alignment-slow", "worsen-alignment-medium", "worsen-alignment-fast",
)
APERTURE_CHANGE_LABELS = (
    "improve-aperture-slow", "improve-aperture-medium", "improve-aperture-fast",
    "stable-aperture",
    "worsen-aperture-slow", "worsen-aperture-medium", "worsen-aperture-fast",
)
# Binary 4-way transitions.
CONTACT_TRANSITION_LABELS = ("gain-contact", "lose-contact",
                             "maintain-contact", "maintain-no-contact")
GRASP_TRANSITION_LABELS = ("gain-grasp", "lose-grasp",
                           "maintain-grasp", "maintain-no-grasp")
SUPPORT_TRANSITION_LABELS = ("gain-support", "lose-support",
                             "maintain-support", "maintain-no-support")
CONTAINMENT_TRANSITION_LABELS = ("gain-containment", "lose-containment",
                                 "maintain-containment", "maintain-no-containment")

# Shared integer encodings for the two generic temporal schemes.
# Continuous 7-way (index meaning is identical across all *_CHANGE heads):
#   0,1,2 = improve(=approach/up/improve) slow,medium,fast
#   3     = stable
#   4,5,6 = worsen(=recede/down/worsen)  slow,medium,fast
CONT_IMPROVE_SLOW, CONT_IMPROVE_MED, CONT_IMPROVE_FAST = 0, 1, 2
CONT_STABLE = 3
CONT_WORSEN_SLOW, CONT_WORSEN_MED, CONT_WORSEN_FAST = 4, 5, 6

# Binary 4-way transition encoding (identical across all *_transition heads):
TR_GAIN, TR_LOSE, TR_MAINTAIN, TR_MAINTAIN_NO = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Eligible pair table (draft §3.2 "Valid pairs" column). Asserted at build time.
# Keys are relation field names; values are sets of allowed (typeA, typeB).
# Object-object is represented as ("object","object"); tcp/gripper are unique.
# ---------------------------------------------------------------------------
ELIGIBLE_PAIRS = {
    "distance":              {("tcp", "object")},
    "height":                {("tcp", "object")},
    "tcp_align":             {("tcp", "object")},
    "aperture":              {("gripper", "object")},
    "contact":               {("gripper", "object"), ("object", "object")},
    "grasp":                 {("gripper", "object")},
    "support":               {("object", "object")},
    "containment":           {("object", "object")},
    # temporal share the absolute pair eligibility
    "distance_change":       {("tcp", "object")},
    "height_change":         {("tcp", "object")},
    "alignment_change":      {("tcp", "object")},
    "aperture_change":       {("gripper", "object")},
    "contact_transition":    {("gripper", "object"), ("object", "object")},
    "grasp_transition":      {("gripper", "object")},
    "support_transition":    {("object", "object")},
    "containment_transition":{("object", "object")},
}

# ---------------------------------------------------------------------------
# Per-relation class counts, for one-hot serialization and aux heads.
# ---------------------------------------------------------------------------
RELATION_NUM_CLASSES = {
    "distance": len(DISTANCE_LABELS),
    "height": len(HEIGHT_LABELS),
    "tcp_align": len(TCP_ALIGN_LABELS),
    "aperture": len(APERTURE_LABELS),
    "contact": len(CONTACT_LABELS),
    "grasp": len(GRASP_LABELS),
    "support": len(SUPPORT_LABELS),
    "containment": len(CONTAINMENT_LABELS),
    "distance_change": len(DISTANCE_CHANGE_LABELS),
    "height_change": len(HEIGHT_CHANGE_LABELS),
    "alignment_change": len(ALIGN_CHANGE_LABELS),
    "aperture_change": len(APERTURE_CHANGE_LABELS),
    "contact_transition": len(CONTACT_TRANSITION_LABELS),
    "grasp_transition": len(GRASP_TRANSITION_LABELS),
    "support_transition": len(SUPPORT_TRANSITION_LABELS),
    "containment_transition": len(CONTAINMENT_TRANSITION_LABELS),
}

RELATION_LABELS = {
    "distance": DISTANCE_LABELS, "height": HEIGHT_LABELS,
    "tcp_align": TCP_ALIGN_LABELS, "aperture": APERTURE_LABELS,
    "contact": CONTACT_LABELS, "grasp": GRASP_LABELS,
    "support": SUPPORT_LABELS, "containment": CONTAINMENT_LABELS,
    "distance_change": DISTANCE_CHANGE_LABELS, "height_change": HEIGHT_CHANGE_LABELS,
    "alignment_change": ALIGN_CHANGE_LABELS, "aperture_change": APERTURE_CHANGE_LABELS,
    "contact_transition": CONTACT_TRANSITION_LABELS,
    "grasp_transition": GRASP_TRANSITION_LABELS,
    "support_transition": SUPPORT_TRANSITION_LABELS,
    "containment_transition": CONTAINMENT_TRANSITION_LABELS,
}
