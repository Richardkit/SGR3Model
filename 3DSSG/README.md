# 3D Scene Graph Framework

This directory contains the 3D scene graph framework used for baseline experiments, including [IMP][IMP], [VGfM][VGfM], [3DSSG][3DSSG], [SGFN][SGFN], and [MonoSSG][MonoSSG]. For the SGR³ pipeline, see [sgr3](../sgr3).

## Contents

- [configs](configs): dataset and method configurations.
- [scripts](scripts): dataset preparation and experiment entry points.
- [data_processing](data_processing): annotation and visibility processing.
- [embedding_and_indexing](embedding_and_indexing): reference feature extraction and indexing.
- [ssg](ssg): model, training, and evaluation implementation.

## Usage

Use [environment.yml](environment.yml) as an environment reference and update dataset paths in the configuration. Obtain 3RScan data under its [terms of use](https://forms.gle/NvL5dvB4tSFrHfQH6).

From this directory:

```bash
python scripts/RUN_prepare_GT_setup_3RScan.py --config /path/to/config.yaml
python main.py --mode train --config /path/to/config.yaml
python main.py --mode eval --config /path/to/config.yaml
```

Implementations of SceneGraphFusion and 3DSSG are adapted within this framework and may differ from their original papers.

## License

This framework is distributed under [CC BY-NC-SA 4.0](LICENSE). The original publications and authors are credited below.

## References
Please cite the original framework papers when using this code:

```
@inproceedings{wu2023incremental,
  title={Incremental 3D Semantic Scene Graph Prediction from RGB Sequences},
  author={Wu, Shun-Cheng and Tateno, Keisuke and Navab, Nassir and Tombari, Federico},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5064--5074},
  year={2023},
}

@inproceedings{wu2021scenegraphfusion,
  title={Scenegraphfusion: Incremental 3d scene graph prediction from rgb-d sequences},
  author={Wu, Shun-Cheng and Wald, Johanna and Tateno, Keisuke and Navab, Nassir and Tombari, Federico},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7515--7525},
  year={2021}
}

@inproceedings{Wald2020,
    title = {{Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions}},
    author = {Wald, Johanna and Dhamo, Helisa and Navab, Nassir and Tombari, Federico},
    booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2020}
}
```
[IMP]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Scene_Graph_Generation_CVPR_2017_paper.pdf
[VGfM]: https://arxiv.org/abs/1807.05933
[MonoSSG]: https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Incremental_3D_Semantic_Scene_Graph_Prediction_From_RGB_Sequences_CVPR_2023_paper.pdf
[SGFN]: https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_SceneGraphFusion_Incremental_3D_Scene_Graph_Prediction_From_RGB-D_Sequences_CVPR_2021_paper.pdf
[3DSSG]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Wald_Learning_3D_Semantic_Scene_Graphs_From_3D_Indoor_Reconstructions_CVPR_2020_paper.pdf
