# 当前论文脚本整理版：排除同场景 rescan，全局参考帧 Top-K

这是 2026-11-01 按论文截图整理的独立代码包。原始脚本、历史结果、正在运行的实验和论文文件均未改动。这里的“旧版”指 **跨场景累计参考帧分数的检索算法，不是先选择一个场景的 argmax 版本**。生成与匹配使用当前本地 Qwen3-VL-32B-Instruct / vLLM 实现；没有恢复未经确认的历史提示词。

## 1. 固定行为

| 环节 | 本包设置 |
|---|---|
| 测试集 | 固定 47 scans，1,503 GT 物体，10,125 条正 GT 关系 |
| 场景隔离 | 根据官方 `3RScan.json` 的 reference/rescan 分组，排除全部测试场景家族；不按 UUID 前缀判断 |
| 窗口 | 6 张图，步长 4；前两个窗口不筛选 |
| ColQwen | ColQwen2.5，阈值 0.5；筛选图用于检索 |
| 查询 patch | SigLIP2；每个 patch 检索 **k=10** 个邻居 |
| 加权 | patch 唯一性权重，温度 **0.5** |
| 参考选择 | 跨查询图累加每个参考帧的得分，跨场景全局取最多 **5 个参考帧**；没有场景 argmax |
| Qwen 图像 | 有保留图时输入该窗口全部原图；全窗口被筛除时跳过检索和生成 |
| Qwen | 本地 Qwen3-VL-32B-Instruct，temperature=0，非推理模式；同一进程中生成、匹配及不同 scan 共用一个模型实例 |
| 输出处理 | 保留现有软解析、词表外名称、可见物体对补边、跨窗口合并和候选排序；在最终评估入口统一截断三个谓词 |
| 评估 | 类别同义词 v2，两个端点类别均正确，关系集合 set@3；谓词语义裁判关闭 |

**截图参数与存档的区别：**目前那份旧式全局参考帧 SGR³ 重测使用 `patch_topk=5`，截图写的是 10。本包按截图固定为 10，并使用全新的检索缓存目录。旧 k=5 缓存、scene-argmax 缓存均不能混用。调整 ANN 查询 k 不要求重新训练已有 FAISS 索引，但必须重新计算窗口参考检索。

生成端当前保留了 `visible_pairs_v1` / `missing_pairs_keep_original_v3` 补边和编号谓词输出。这是现有重测实现，不应仅凭这个整理包宣称它与所有历史论文运行的提示词完全相同。`reports/SOURCE_ORIGINS.json` 记录了实际来源。

## 2. 脚本入口

只从 `tools/` 运行；`src/`、`legacy_matching/` 是内部实现。

| 入口 | 用途 | 资源 |
|---|---|---|
| `tools/prepare_config.py` | 展开配置路径并检查论文参数，写入新配置文件 | CPU |
| `tools/audit_kb.py` | 官方场景分组隔离检查；可加 `--actual-keys` 检查实际索引键 | CPU |
| `tools/precompute_retrieval.py` | 用现有索引和已缓存特征，一次完成全部窗口的 k=10 参考检索 | CPU，需 FAISS 和较多内存 |
| `tools/run_local.py` | 本地 Qwen 生成、实例匹配、统一评估；一个模型实例贯穿本次运行 | GPU；当前实现适合单 H200 |
| `tools/evaluate.py` | 对已有图和物体对应重新计分，不调用模型 | 纯 Python / CPU |
| `tools/verify_archives.py` | 对项目已有主表和消融存档做回归核验 | CPU |
| `tools/preprocess.py` | 可选：重新计算 ColQwen 决策和 SigLIP 查询特征 | GPU |
| `tools/embed_reference.py` | 可选：沿用旧 4×6 SigLIP 编码方法重算参考特征，先排除测试场景家族 | GPU |
| `tools/build_kb.py` | 可选：从现有参考特征另建场景隔离索引 | CPU |

关键实现：`src/vision_legacy.py` 是加权 patch 检索；`src/filter_legacy.py` 是 ColQwen；`src/graph_legacy.py` 是窗口和合并；`src/run.py` 是生成；`src/evaluate_single_scan_local.py` 是物体对应；`evaluation/evaluate.py` 是最终评分。

`src/dashscope_client.py` 仅保留原 JSON 解析函数，以兼容导入名，已移除 HTTP/API 客户端。这个包不需要 DashScope key。

## 3. 当前工作区使用方法

先设置脚本包路径、已有环境和模型位置。下面的 Python 路径是整理时可访问的环境，没有新安装包。`requirements-observed.txt` 只记录该环境的实际版本，不是可直接安装的完整锁文件。FAISS 检索需另外使用已安装 FAISS 的环境；评分只需 Python 标准库。

```bash
SGR3_PACKAGE='/hfs2/work/workspace/scratch/ka_rg5174-rag_migration/sgr3'
SGR3_PYTHON='/hkfs/work/workspace/scratch/rg5174-rag4_recovered/rg5174-ragspace4-1791422104/rg5174-ragspace-1780103764/envs/rag_env/bin/python'
QWEN_MODEL_PATH='/hkfs/work/workspace/scratch/rg5174-rag4_recovered/rg5174-ragspace4-1791422104/rg5174-ragspace-1780103764/models/Qwen3-VL-32B-Instruct'
export SGR3_PACKAGE SGR3_PYTHON QWEN_MODEL_PATH
```

生成一个自己的配置文件。目录迁移时加 `--project-root /实际项目根目录`，并核对配置内的数据、模型缓存和索引路径。

```bash
"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/prepare_config.py" --output /新工作目录/paper_config.json
"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/audit_kb.py" --config /新工作目录/paper_config.json
"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/run_local.py" --config /新工作目录/paper_config.json
```

没有 `--execute` 时，模型运行、预处理、embedding、建库和预检索入口只检查/打印计划，不进行模型推理。若预检报告 `needs_new_k10_retrieval_cache`，先运行下面的预检索命令生成缓存。

实际运行应先在 CPU 资源上完成参考检索，再在 GPU 资源上测试一个 scan。示例中的绝对路径占位符需要替换。

```bash
/具有FAISS和torch的环境/bin/python "$SGR3_PACKAGE/tools/precompute_retrieval.py" --config /新工作目录/paper_config.json --execute
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"$SGR3_PYTHON" "$SGR3_PACKAGE/tools/run_local.py" --config /新工作目录/paper_config.json --model-path "$QWEN_MODEL_PATH" --output-root /新工作目录/pilot --scans 一个测试scan的UUID --execute
```

去掉 `--scans` 才会处理全部 47 scans。正式输出必须使用新目录；不会覆盖旧结果。这个简单入口按 scan 顺序执行，不自动申请或续订 Slurm，也不自动恢复中断目录；长实验可分成多个 scan 子集，保留各自预测后汇总到同一评估 manifest。未完成的 scan 仍在 New 分母内，部分结果明确标记为非最终结果。

生成前从已封存的检索缓存读取图片和参考边，不加载 FAISS。`reports/model_path.txt`、运行配置、源码副本、预检记录、逐窗口响应与最终图都保存在新运行目录。模型初始化一次的检查贯穿每个 scan。

## 4. 新 eval：精确定义与独立使用

固定协议名：`robio-category-equivalence-directed-set-v4`；别名文件：`evaluation/object_aliases_v2.json`。

1. 使用有效 ID 和一对一 GT→预测实例对应。物体类别采用固定同义词表，包含已确认的 `chair/armchair`、`table/desk`、`heater/radiator` 等；不临时调用语义模型猜同义词。
2. 同一有向物体对的谓词按导出顺序合并、归一化、去重，再取前三个非 none 候选。不会把无关系边计作正关系命中。
3. `H`：两个端点都已对应且类别正确，并且 GT 谓词命中前三候选的 **GT 关系标注条数**。同一对多个 GT 谓词分别计数。
4. `D_Old`：两个端点已对应且类别正确、预测对有非空候选的 GT 关系标注条数。`Old = H / D_Old`。
5. `D_New`：固定测试名单的全部正 GT 关系，完整 47 scans 时为 10,125。`New = H / D_New`。漏物体、漏关系及失败 scan 不从 New 分母删除。
6. `New mRecall`：对这份固定测试集上有 GT 的 25 个谓词类别，分别算 `H_r / D_New,r`，再等权平均。零命中类别保留。命中同样要求两个端点类别正确。
7. `object_category_recall_after_association`：类别正确且对应成功的 GT 物体数 / 全部 GT 物体数。它不是 GNN 分类器 Object R@10，也不是不检查类别的实例对应率。
8. `node_ratio`：先汇总所有预测节点数，再除以所有 GT 节点数。它不受类别是否正确影响。

**set@3 不标为联合 R@1。**GNN、其他 baseline 和 SGR³ 在同一图/对应格式上使用同一个评分入口。现有 GNN 导出规则（预测 Top1 物体类别，谓词阈值后按分数排序）并未由本包偷偷修改。原生 GNN Object R@10、Predicate R@3 是另外的边际指标，不从这里的关系命中率替代生成。

单个 prediction 文件示例：

```json
{"nodes":[{"id":10,"name":"armchair"},{"id":20,"name":"desk"}],"edges":[{"src":10,"tgt":20,"c_edge":["left","close by","front"]}]}
```

mapping 示例：`{"1":"10","2":"20","3":"none"}`，方向是 GT ID → 预测 ID。也接受 `{"gt2pred":{...}}` 外层结构；旧 JSONL 需先按 scan 整理为该格式，不能直接作为 mapping 传入。

评估 manifest 示例（实际应列全 47 个 scan；缺项作为 pending，不偷偷缩小分母）：

```json
{
  "method":"SGR3_paper",
  "split":"/实际路径/evaluation_scans.txt",
  "GT_root":"/实际路径/data/3RScan",
  "scans":{
    "一个测试scan的UUID":{
      "state":"complete",
      "prediction":"/实际路径/graph_final.json",
      "mapping":"/实际路径/object_matching.json"
    }
  }
}
```

```bash
python "$SGR3_PACKAGE/tools/evaluate.py" --manifest /实际路径/manifest.json --output /新结果路径/scores.json
```

生成流程会自动写出这种 manifest 和分数。`runs/.../object_matching/results.json` 是匹配脚本保留的**旧诊断输出**；论文最终值应读取新入口产生的 `reports/manifest_XX_scores.json`，不要混用。

## 5. 数据和缓存依赖

本包只包含代码、固定 split/词表/别名及核验记录，不包含模型、RGB、GT、embedding、索引、大型特征缓存、旧预测或凭据。当前配置引用现有的场景隔离 KB 和已经完成的 ColQwen/SigLIP 查询特征缓存。即使 embedding 根目录中含有旧测试扫描，新建库入口仍按官方场景家族排除。

可选预处理入口仍依赖每个测试 scan 已有的 `siglip2_val/patch_tokens_fp16.npy` 和 `paths.json` 作为筛选候选的 proxy；没有把这些既存数据的历史生产过程声称为已重新复现。默认无需重算它们。

若确需重算参考 embedding/索引，用 `embed_reference.py`、`build_kb.py`，指定全新输出目录，并在配置里明确指向新产物。若重算测试特征，使用 `preprocess.py`，完成全部 47 scans 后再把 `precomputed_cache_dir` 指向它；单独 shard 不能冒充完整缓存。
