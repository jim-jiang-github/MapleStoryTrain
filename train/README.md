# WzToolkit 怪物识别 YOLO 训练管线

一键从 WZ 数据训练「怪物种类」识别模型，全程脚本化、可复现。

## 目录结构

```
train/
├── run.bat                # 本地一键全流程（round1 编排，CPU 训练用）
├── 训练监控.bat            # 远程服务器训练实时监控（进度条/倒计时）
├── README.md              # 本文件
├── src/                   # ★ 所有训练代码
│   ├── gen_data.py        # ★ 训练图生成统一入口（sample/full/weak/validate/review）
│   ├── batch_export.py    # round1 出图引擎（log/tile_windows 公共库）
│   ├── batch_export2.py   # round2 出图引擎（取景框 1:1 · 只标怪 · 疏密两版）
│   ├── boost_weak.py      # round1 弱类补数据
│   ├── boost_weak_r3.py   # round3 弱类定向加量
│   ├── compact_classes.py # 压缩类别表（只留出现过的类）
│   ├── overnight_run.py   # round1 过夜编排（导出→切分→训练→弱类→报告）
│   ├── prep_split.py      # train/val 8:2 切分
│   ├── predict_test.py    # 全新测试图终验
│   ├── review_gallery.py  # 自检页生成（review.html）
│   ├── validate_r2.py     # 数据全量体检
│   ├── package_kit.py     # 打包训练包（发别的电脑）
│   ├── finetune.py        # ★ 服务器微调脚本（kit 内自带，imgsz=1366 + rect）
│   ├── export_onnx.py     # ★ 导出 yololab 用 onnx（动态 shape + 冒烟核对张量）
│   ├── upload_kit.py / upload_kit_par.py   # SFTP 上传（单流 / 8 路并行）
│   ├── upload_delta.py    # 增量上传新样本
│   ├── live_view.py       # 远程训练实时监控
│   ├── monitor_epoch.py   # epoch 速度记录
│   └── archive/           # 一次性修复脚本（reexport_all / reexport_noboss）
├── data/                  # ★ 生成的训练数据
│   └── yolo_r2/           # 当前数据集：images/labels + classes + dataset.yaml + review/
├── runs/                  # 训练产物
│   ├── r1/weights/best.pt # round1（本地 CPU 100 epochs）
│   └── r2/weights/best.pt # round2（4090 100 epochs + 弱类补数据微调 30 epochs）
├── kit/                   # 打包的训练包（wz_mob_trainkit.zip，13GB）
├── logs/                  # 全部历史日志
└── reports/               # 结果表：perclass_*.txt / weak_classes.txt / batch_*_result.json
```

`exports/`（项目根）只留 `maps/`——那是 wztool 地图渲染的通用输出，与训练无关。

## round2 数据集：全量练级图 · 取景框 1:1 · 只标怪物

> **2026-11-06 数据集已清理**（自动生成物不入库，需要时按下文命令重新生成，全流程约 1 小时）：
>
> ```bat
> python train\src\gen_data.py full --workers 8   :: 导出（约 1 小时）
> python train\src\gen_data.py weak               :: 弱类补数据（可选）
> python train\src\prep_split.py 0.2              :: train/val 切分
> python train\src\gen_data.py validate           :: 全量体检
> python train\src\gen_data.py review --sample 400 :: 抽样自检
> python train\src\package_kit.py                 :: 需要发车训练时再打包
> ```
>
> 清理前终检快照（17450 张 · 0 损坏 · 0 标注错位 · 0 刷怪不符）：图片 17450、
> 怪框 201059、类 473、纯背景负样本 1976（12.2%）。

**历史数据规格**：12657 张基础 + 1212 弱类补数据 · 710 张地图 · 469 类，全部 1366×768。

| 校验项 | 结果 |
|---|---|
| 图片/标签一一对应 | 0 缺 |
| classes.json / classes.txt / dataset.yaml | 三方一致（469） |
| 框坐标（字段数/越界/过小/巨框） | 0 / 0 / 0 / 0 |
| PNG 完整性（全量 verify） | 0 损坏 |
| **标注的怪是否真刷在这张地图**（life + link 链） | **0 不符** |
| 残留 BOSS 级类别 | 0 |
| 纯背景负样本 | 1545 张（12.2%） |

已知取舍：27 张纯 BOSS 图整张不导（符合「只要小怪」）；十几张超大图
（单窗口渲染 5~10 分钟）有零星瓦片超时未出，地图覆盖率仍是 710/710。

## 快速开始：出图统一走 gen_data.py

```bat
python train\src\gen_data.py sample --maps 5 --dry-run  :: 先看规模
python train\src\gen_data.py sample --maps 5            :: 小批试跑
python train\src\gen_data.py full --workers 8           :: 全量导出（约 1 小时）
python train\src\gen_data.py weak --dry-run             :: 弱类定向补数据
python train\src\gen_data.py validate                   :: 数据全量体检
python train\src\gen_data.py review --sample 400        :: 抽样自检页（全量去掉 --sample）
```

**必须在项目根目录运行**（`wztool.exe` 在根目录）。

本地一键全流程（round1 编排）：

```bat
train\run.bat                          :: 默认参数
train\run.bat --maps 500 --epochs 150  :: 自定义参数
```

和 round1（`batch_export.py`）的四个关键区别：

| | round1 | round2 |
|---|---|---|
| 出图方式 | `--region` 从整图缩略图裁剪，大地图被压到 ~0.53x | `--view` 按相机**1:1 渲染**（背景走客户端视差公式 + 自动补云层），和真实截图同像素尺度 |
| 标注 | 怪 + 绳/梯 + 平地 + 斜坡 | `--mob-only`：**只标怪** |
| 类别 | `id_名字__动作状态`（386 类，样本被切碎） | 物种级 `id_名字`（520 类，动作只做样本多样性） |
| 每瓦片 | 1 个样本 | **疏（density 1）+ 密（density 3）两份**，不同 seed → 不同怪组合/朝向/动作 |

细节：

- **网格覆盖**：1366×768 瓦片，相邻重合 1/3（步长 911×512），末块贴地图边缘。
  纯天空/没刷怪点的瓦片自动跳过（判据：瓦片内 foothold 覆盖 < 120px 且无刷怪点）。
- **怪物合成**：每个刷怪点按其 `rx0~rx1` 巡逻区间撒怪，与取景框求交；`foothold`
  线性插值落地（不悬空）；50% 随机镜像朝向；动作按权重抽（站/走/飞多，攻击次之，
  受击/死亡/跳跃少），动作内随机抽帧；单张上限 `--max-mobs 30`。
- **类别表预注册**：`wztool yolo-classes` 先把全部会出场的怪按 id 排序写进
  `classes.json`，导出时只读。8 路并发不再写注册表（以前用锁文件，Windows 上
  反复创建同名点文件会被安全策略拦，导致全体卡死）。
- **导出必须设 `WZ_CLASSES_RO=1`**（重要）：预注册漏掉的类会让每个导出进程
  各自 `读-改-写` classes.json，`id = len(cur)` 在同一位置互相覆盖——先写那批
  图的 label 就指向了别人的类。round2 首版就是这样废掉 2569 张（162 张地图）：
  标签写着「木桩靶子」，框里其实是「龙」。
  现在 `WZ_CLASSES_RO=1` 下缺类会直接报错，绝不会静默写坏。
- **必做的终检**：`python train\src\gen_data.py validate`——除了图片/标注/类别表/PNG
  完整性，还会校验「每个标注的怪必须真的刷在这张地图的 life 里」（含 link 链），
  这条能抓出所有 id 串号。
- **断点续跑**：同 `--tag` 的任务计划确定，重跑自动跳过已存在的图。
- 产出：`train/data/yolo_r2/{images,labels}/`、`classes.txt`、`dataset.yaml`、
  `review.html`（带框缩略图 + 类别分布，单文件可直接看）。

## 远程训练（当前主用流程）

1. **打包**：`python train\src\package_kit.py` → `train/kit/wz_mob_trainkit.zip`（含数据+权重+finetune.py+dataset.yaml，跨机相对路径）
2. **上传**：`python train\src\upload_kit_par.py`（8 路分块并行，13GB 约 35 分钟）
3. **服务器训练**：AutoDL 4090，`nohup /root/miniconda3/bin/python finetune.py &`（kit 内自带）
4. **实时监控**：双击 `train\训练监控.bat`（进度条/倒计时/最近 mAP，自动跟随最新日志）
5. **增量补数据**：`python train\src\upload_delta.py`（只传新增样本，`reports/delta_manifest.txt`）

当前模型（runs/r2，val mAP50=0.914）已部署：`yololab/models/best_ft.onnx`。

## 训练分辨率：1366×768 原生，别再回 960

2026-11-07 改的。以前 `imgsz=960` 是**方形画布**，加载器会把 1366×768 的信箱缩放
先缩到 0.70 倍再上下补 210px 灰边——最小的怪（蜗牛 35px）实际只剩 25px，
同时 44% 的算力花在灰边上。

现在统一 `imgsz=1366 + rect=True`：

| | 旧 imgsz=960 | 现 imgsz=1366 + rect |
|---|---|---|
| **训练**批次张量 | 960×960（内容仅 960×540） | 800×1376（内容满幅） |
| 训练缩放比 | 0.703 | 1.007（1:1） |
| 35px 蜗牛 | 24.6px | 36px |
| 训练像素量 | 0.92M | 1.10M（+19%） |
| 推理（实机实测 CPU） | 960×544 → 中位 92 ms | 800×1376 → 中位 236 ms |

推理那行要注意：旧版 `imgsz=960` 推理其实**已经**走 rect（ultralytics 预测器默认
`rect=True`），喂进去是 960×544；新版多了 2.1 倍像素，CPU 上 10fps → 4fps。
所以覆盖层建议 `--device cuda:0`；只能用 CPU 就把下拉切到 1152（约 163 ms），
别切回 960（尺度差 30%，正好把这次的收益亏回去）。

改到的位置（保持一致，别只改一处）：

- `src/package_kit.py` → 生成的 `train.py`：`IMGSZ=1366, RECT=True`
- `src/finetune.py`、`src/overnight_run.py`（`--imgsz` 默认 1366）
- `src/predict_test.py`、`src/predict_test_r2.py`：`imgsz=1366, rect=True`
- `yololab/app.py`（API/实时默认值 + `predict(rect=True)`）、
  `yololab/static/index.html`（下拉默认 1366）、`yololab/overlay.py`（`--imgsz` 默认）

**已知代价**：ultralytics 在 `build_transforms` 里对 `rect=True` 会强制把
mosaic / mixup / cutmix 置 0，所以这批训练没有 mosaic。想要 mosaic 就把 `RECT=False`
（画布变 1376×1376，算力约 2 倍）。

**导出 ONNX 必须动态 shape**，否则 rect 推理失效（会退回 1376×1376 补灰边）：

```bat
python train\src\export_onnx.py            :: imgsz=1366 + dynamic，自动冒烟核对张量
```

正常应打印 `实际喂入张量 (1, 3, 800, 1376)`。旧的 `best_ft.onnx`（960 训练的）
与新模型尺度不一致，必须重导。

## 依赖

- `pip install ultralytics`（会自动带 torch CPU 版）
- 远程训练需要 GPU 服务器（AutoDL 等），kit 内含环境安装说明
- 上传脚本依赖 `pip install paramiko`

## 产物位置

- `train/runs/r1|r2/weights/best.pt` 模型
- `train/reports/perclass_*.txt` 逐类评估表、`batch_*_result.json` 导出结果
- `train/data/yolo_r2/` 数据集（images/labels + classes.json + dataset.yaml）
- `train/kit/wz_mob_trainkit.zip` 可发车训练包
- `train/logs/` 全部历史日志

## 历史备注

- 2026-11-06：目录整理——代码集中到 `src/`，数据集移入 `data/`，训练产物移入
  `runs/`，日志/报表归类；出图入口统一为 `gen_data.py`。
- 2026-11-05 之前管线脚本放在 `.cache/_check/`（会被 `wztool clear-cache` 连缓存
  一起删掉），已迁入本目录并进 git。
