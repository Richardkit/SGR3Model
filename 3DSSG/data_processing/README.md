# Data Processing

Scripts for preparing annotations, visibility graphs, and image crops. Dataset preparation entry points are listed in the [framework README](../README.md).

## Proposal Format

`proposals.h5` stores objects and keyframes for each scan:

```text
<scan_id>
  nodes/<object_id>   keyframe indices; label attribute
  kfs/<keyframe_id>   bounding box rows; seg2idx attribute
```

Each bounding box row contains `[x1 / width, y1 / height, x2 / width, y2 / height, occlusion]`. The `seg2idx` attribute maps an object or segment ID to its bounding box row index. For estimated segments, node IDs refer to segments rather than GT object instances.
