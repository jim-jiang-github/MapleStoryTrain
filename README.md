# MapleStoryTrain

MapleStory 怪物检测（YOLO）训练仓库：数据生成 → 训练 → 导出 ONNX → 部署验证。

地图/怪物的**渲染引擎**不在本仓库，在 [`vendor/WzToolkit`](https://github.com/jim-jiang-github/WzToolkit)（git submodule）——本仓库只负责训练侧策略：标注规则、类别表、数据划分、训练、部署。

## 目录结构

```
MapleStoryTrain/
├── vendor/WzToolkit/     ← 地图导出工具（git submodule，渲染+随机摆怪引擎）
├── train/
│   ├── src/
│   │   ├── gen_data.py       统一入口：sample / full / weak / validate / review
│   │   ├── batch_export2.py  全量导出（网格全覆盖、疏密两版、负样本）
│   │   ├── yolo_export.py    渲染走子模块 + YOLO 标注（训练策略都在这）
│   │   ├── wztk.py           子模块加载器（把 WzToolkit 当库用）
│   │   ├── prep_split.py     train/val 划分
│   │   ├── package_kit.py    打包训练 kit（发给 GPU 机器）
│   │   ├── finetune.py       GPU 上的训练脚本（kit 自带）
│   │   ├── export_onnx.py    best.pt → ONNX（--static 固定 800×1376）
│   │   └── ...
│   ├── data/yolo_r2/     数据集（images/labels 不进 git，可由 gen_data 重新生成）
│   ├── reports/          历史评估报告（per-class 排名、弱类清单）
│   └── README.md         训练管线详细说明（含训练分辨率章节）
└── yololab/              部署侧：Web 推理 + 游戏内叠加层（models/ 不进 git）
```

## 首次使用

```bash
git clone https://github.com/jim-jiang-github/MapleStoryTrain.git
cd MapleStoryTrain
git submodule update --init --recursive      # 拉取 WzToolkit（含 bin/*.exe）

# 1. 告诉训练管线 .wz 转出来的 nx 文件在哪（写进 workspace.json，不进 git）
python -c "import sys; sys.path.insert(0,'train/src'); import wztk; wztk.set_workspace(r'C:\path\to\nxrun')"

# 2. 小批试跑（先看规模和画面对不对）
python train\src\gen_data.py sample --maps 3

# 3. 全量生成（本机 CPU，1-2 小时；疏 d1 + 密 d3 两版 + 负样本）
python train\src\gen_data.py full --workers 8

# 4. 划分 + 打包 + 上 GPU
python train\src\prep_split.py 0.2
python train\src\package_kit.py
# → 上传 kit 到 4090 跑 finetune.py（约 3 小时）
# → 拿回 best.pt 后：python train\src\export_onnx.py --static
```

## 架构：为什么标注在训练仓库，不在 WzToolkit

上游 `wztool map-export` 的职责是**出图**（渲染 + 沿 foothold 随机摆怪）。它不认
`--yolo / --mob-only / --no-boss / --max-mobs` 这些参数——「标什么、标多少」是
训练策略，不该长在地图工具里。所以 `wztk.py` 把子模块当**库**加载：

| 环节 | 谁负责 | 说明 |
|---|---|---|
| 地图解析 | 子模块 `nxmap.exe` | map.json + 贴图，缓存进本仓库 `.cache/` |
| 渲染 | 子模块 `_render_preview` | 1:1、客户端同款层序/视差，与 exe 同一份代码 |
| 摆怪 | 子模块 `_composite_mobs` | 贴地、动作加权随机、seed 可复现，返回每只怪像素框 |
| 窗口 | 本仓库 | 裁 1366×768（游戏画面固定尺寸，不缩放） |
| 标注 | 本仓库 | YOLO 框、classes.json（id 永不漂移）、no-boss/max-mobs 策略 |

好处：上游更新只要 `git submodule update --remote vendor/WzToolkit`，不用 merge fork。

## 训练分辨率（重要）

训练/推理均为 **imgsz=1366 + rect=True**：实际张量 800×1376，缩放比 1.007，
无灰边，35px 的小怪保持原尺寸。详见 `train/README.md` 的「训练分辨率」章节。
部署侧 `yololab` 默认推理尺寸与训练一致；静态 ONNX（`export_onnx.py --static`）
只接受 800×1376。

## 与旧 worktree（WzToolkit/dev-5db013a8）的关系

本仓库是训练管线的**唯一维护地**，旧 worktree 的 `train/` 已迁移至此。
WzToolkit 仓库继续只做地图工具，两边通过 submodule 解耦。
