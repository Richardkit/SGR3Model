# SGR³ Model: Scene Graph Retrieval-Reasoning Model in 3D

Research code for retrieval-augmented 3D scene graph prediction from RGB sequences.

## Getting Started

The organized SGR³ implementation is in [sgr3](sgr3). Its [README](sgr3/README.md) describes the configuration, dependencies, execution steps, and evaluation protocol.

## Repository Structure

- [sgr3](sgr3) — SGR³ pipeline with generation, object matching, retrieval, and shared evaluation entry points.
- [3DSSG](3DSSG) — 3D scene graph data processing, feature extraction, and knowledge-base indexing.
- [RAG_baseline_Qwen3](RAG_baseline_Qwen3) — Qwen3-VL baseline methods and scene graph evaluation scripts.
- [RAG_agentic](RAG_agentic) — SGR³ scene graph prediction, frame filtering, retrieval, and representative ablation implementations.
- [RAG_Rexplore_Qwen3](RAG_Rexplore_Qwen3) — Experience abstraction and window-level retrieval experiments.
