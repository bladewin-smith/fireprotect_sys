# YOLO26 + Orbbec Gemini 336L 火焰识别与 RGB-D 定位

这是一个可直接继续开发的 Windows 优先项目，覆盖以下闭环：

1. Gemini 336L 同步采集彩色图、对齐深度图和相机内参；
2. 使用已有 YOLOv5 `fire/smoke` 模型生成自动框，并逐张人工接受、否决或丢弃；
3. 按场景划分 YOLO 数据集并检查标签；
4. 使用 YOLO26 预训练模型微调火焰检测器并在独立测试集评估；
5. 在 RTX 3060 12GB 上以 PyTorch 或 TensorRT 实时推理；
6. 将二维火焰框与对齐深度结合，输出相机坐标系下的距离和 XYZ 代理位置。

> 重要限制：主动双目/结构光相机通常不能直接测到半透明、强发光且不断变化的火焰表面。项目默认在检测框底部的小区域取稳健中值，更接近“燃烧物或火焰后方表面”的距离，而不是火焰本体的精确三维位置。事件日志中的 `depth_is_proxy` 因此固定为 `true`。若项目要求计量级火焰三维边界，应增加热成像、双目多视角或专门的火焰物理传感器。

## 1. 检测与定位原理

YOLO26 是单阶段目标检测器：一帧 RGB 图像经过特征提取和多尺度融合后，检测头直接输出类别置信度及边界框。这里通过迁移学习，把 COCO 预训练权重中的通用视觉特征微调为一个 `fire` 类。YOLO26 检测模型同时训练一对多和一对一检测头：默认推理使用一对多头加 NMS，导出时也可以选 `nms=False` 的端到端一对一头。

一次完整检测大致是：

1. 将输入按比例缩放并填充到 `imgsz × imgsz`，归一化后送入网络；
2. 主干提取从纹理、边缘到高层语义的特征，颈部融合不同分辨率特征，因此同一模型能兼顾近处大火焰和远处小火焰；
3. 检测头在多个尺度上回归候选框并预测 `fire` 置信度；训练时分类与框回归损失共同更新参数；
4. 常规一对多头经过置信度过滤和 NMS 去掉高度重叠的重复框；端到端一对一头则可直接输出有限数量的最终框；
5. Ultralytics 把填充后坐标映射回原图，本项目随后才在原分辨率对齐深度图中取样。

模型只用 RGB 图训练；深度不送入 YOLO。推理时先检测二维框，再在已经映射到彩色相机视角的深度图中取样。对彩色像素 `(u, v)` 和深度 `Z`，用针孔模型计算：

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth
```

输出坐标系为相机坐标系：`+X` 向图像右侧，`+Y` 向下，`+Z` 从相机向前。代码使用框底部中央 ROI、有效范围过滤、中值和 MAD 离群值过滤，降低深度孔洞和飞点影响。

## 2. 项目结构

```text
yolov26_fire_rgbd/
├─ configs/fire.example.yaml       # 数据 YAML 示例
├─ data/raw/                       # 按场景保存原始 RGB-D 和标签
├─ data/fire/                      # prepare_dataset.py 生成的训练集
├─ scripts/
│  ├─ collect_rgbd.py              # Gemini 336L 数据采集
│  ├─ pseudo_label_yolov5.py       # 调用旧 YOLOv5 教师模型生成隔离的自动标签
│  ├─ review_pseudo_labels.py      # 人工接受/否决自动标签，提升为正式标签
│  ├─ prepare_dataset.py           # 按场景拆分/复制数据
│  ├─ check_dataset.py             # YOLO 标签检查
│  ├─ train.py                     # YOLO26 微调
│  ├─ evaluate.py                  # 在冻结的 val/test 划分上评估并保存 JSON 摘要
│  ├─ export_model.py              # ONNX / TensorRT 导出
│  ├─ infer_media.py               # 图片、视频、普通摄像头二维推理
│  └─ infer_rgbd.py                # Gemini 实时检测 + 深度定位
├─ src/fire_rgbd/                  # 相机、深度估计和可视化公共代码
├─ tests/                          # 不依赖相机的单元测试
├─ weights/                        # 可手工放置预训练/部署权重
└─ runs/                           # 训练、推理和事件日志
```

以下命令都假定 PowerShell 当前目录是本项目根目录：

```powershell
cd D:\GLM\Master\fireprotect_sys\yolov26_fire_rgbd
```

## 3. 硬件和安全准备

- Gemini 336L 尽量直连主板 USB 3.x 端口，不要先接无供电 Hub。
- 用 `nvidia-smi` 确认 RTX 3060 和驱动可见；若 CUDA 版 PyTorch 无法加载，先更新 NVIDIA 驱动。
- 推荐 Python 3.11 x64。虽然依赖支持更高版本，3.11 通常与视觉、TensorRT 工具链兼容得更稳。
- 真实明火采集必须在获准的受控区域，由具备消防知识的人员操作，配置灭火器/灭火毯和监护人。不要为了扩充数据自行在室内、林地或设备附近点火。可优先使用已获许可的消防演练、公开录像和安全的小型标准火源。

## 4. 安装环境

### 4.1 创建虚拟环境

如果尚未安装 Python 3.11，请从 Python 官网安装 x64 版本并勾选 Python Launcher。然后执行：

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 4.2 安装 CUDA 版 PyTorch

RTX 3060 可使用官方 CUDA 12.6 wheel；wheel 自带 CUDA 运行时，不要求另装完整 CUDA Toolkit：

```powershell
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
```

如果显卡驱动太旧，请先更新驱动；也可根据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 换成当时支持的 CUDA wheel。不要误装 CPU-only 的 `torch`。

### 4.3 安装项目和 Orbbec SDK

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

本项目使用 Orbbec SDK v2 的 PyPI 包 `pyorbbecsdk2`，导入名仍然是 `pyorbbecsdk`。首次安装后，建议以管理员 PowerShell 运行一次官方环境配置脚本：

```powershell
$orbbecDir = python -c "import os,pyorbbecsdk; print(os.path.dirname(pyorbbecsdk.__file__))"
python "$orbbecDir\shared\setup_env.py"
```

拔插相机，然后验证：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "from pyorbbecsdk import Context; print('Orbbec devices:', Context().query_devices().get_count())"
```

第一条应显示 `True` 和 RTX 3060，第二条设备数应至少为 1。若 SDK 报固件不兼容，先用 Orbbec 官方 Viewer 检查设备和固件，不要盲目刷写固件。

模型 `yolo26n.pt`/`yolo26s.pt` 第一次使用时会自动联网下载。离线电脑可提前在联网电脑下载权重，复制到 `weights/`，训练时传 `--model weights/yolo26s.pt`。

## 5. 采集数据

### 5.1 先设计场景，而不是只追求帧数

建议初版至少采集 20～30 个相互独立的场景；一个固定机位连续拍摄的几百帧仍只算一个场景。目标可先定为 2,000～5,000 张经过筛选的图，其中 30%～50% 是无火负样本或困难负样本。

需要覆盖：

- 小火、中火、画面边缘火焰、部分遮挡、近中远距离；
- 白天、夜间、逆光、弱光、不同曝光和室内外背景；
- 稳定机位与轻微运动、不同俯仰角；
- 红色/橙色灯光、太阳、反光、焊接光、屏幕火焰、蒸汽、烟雾等容易误报的负样本；
- 实际部署地点的无火长视频，这是抑制误报最有价值的数据之一。

不要把训练、验证、测试都采自同一段连续视频。应让验证/测试包含不同时间、机位或地点。

### 5.2 运行采集器

手动模式，按 `S` 保存当前一组 RGB-D，按 `A` 切换自动采集，按 `Q` 退出：

```powershell
python scripts/collect_rgbd.py --scene lab_day_fire_01
```

每 0.5 秒自动保存，采够 300 组后退出：

```powershell
python scripts/collect_rgbd.py --scene lab_day_fire_02 --auto --interval 0.5 --max-samples 300
```

无火困难负样本也要单独建场景：

```powershell
python scripts/collect_rgbd.py --scene lab_red_light_negative_01 --auto --interval 1.0 --max-samples 200
```

程序优先使用硬件 D2C。若相机/固件不支持或画面对齐异常，可显式使用软件对齐：

```powershell
python scripts/collect_rgbd.py --scene align_test --software-align
```

每个场景会形成：

```text
data/raw/lab_day_fire_01/
├─ images/   # 训练和标注使用的 RGB JPG
├─ depth/    # 同名 16 位 PNG，像素值单位为 mm，0 表示无效
├─ meta/     # 同名 JSON，含时间、对齐方式和彩色相机内参
├─ labels/   # 后续放同名 YOLO txt
└─ session.json
```

采集后先人工删除严重模糊、过曝、重复度极高或 RGB/Depth 明显错位的样本。深度图仅用于部署定位，不作为本版 YOLO 的训练输入。

## 6. 标注

### 6.1 本项目如何使用已有 YOLOv5 权重

旧工程位于：

```text
D:\GLM\Master\fireprotect_sys\smoke\yolov5-6.0-fire_smoke
```

已核对旧数据配置，类别为：

```text
0 fire
1 smoke
```

默认教师权重使用：

```text
..\smoke\yolov5-6.0-fire_smoke\runs\train\fire_smoke-v5m\weights\best.pt
```

选择 v5m 是因为现有训练记录中，其最后一轮 `mAP@0.5:0.95` 约为 0.930，高于 v5s 的约 0.878。这个数值很高，可能受旧数据集重复帧、场景泄漏或数据难度影响，**只能用于选择预标注教师，不能当作新 RGB-D 数据上的真实性能结论**。新模型必须在按场景隔离的测试集重新评估。

YOLOv5 权重只负责生成候选框，不会被加载进 YOLO26。YOLO26 仍从 `yolo26n.pt` 或 `yolo26s.pt` 开始微调。

### 6.2 生成隔离的自动标签

推荐让包装脚本运行在本项目环境中，再通过 `--python` 指向能够运行旧 YOLOv5 的环境。当前电脑已有的旧环境可以这样调用：

```powershell
python scripts/pseudo_label_yolov5.py `
  --source data/raw `
  --python C:\Users\31866\miniconda3\envs\yolov5\python.exe `
  --yolov5-dir ..\smoke\yolov5-6.0-fire_smoke `
  --weights ..\smoke\yolov5-6.0-fire_smoke\runs\train\fire_smoke-v5m\weights\best.pt `
  --class-map 0:0 `
  --imgsz 416 `
  --conf-thres 0.35 `
  --iou-thres 0.45 `
  --device 0 `
  --half
```

先检查路径和最终命令而不运行模型：

```powershell
python scripts/pseudo_label_yolov5.py --source data/raw `
  --python C:\Users\31866\miniconda3\envs\yolov5\python.exe --dry-run
```

只处理一个场景可加 `--scene lab_day_fire_01`。默认 `--class-map 0:0` 表示只保留旧模型的 `fire`，映射为新数据集类别 0；旧模型检测到的 `smoke` 被忽略。脚本会为每个场景新增：

```text
data/raw/<scene>/
├─ labels_auto/       # 五列 YOLO 自动标签；包括同名空负样本标签
├─ predictions_auto/  # 每张图的教师权重、置信度和框 JSON
└─ review_auto/       # 旧模型画框后的预览图
```

运行级参数、教师权重 SHA-256、阈值和逐图统计保存在：

```text
runs/pseudo_label_yolov5/<timestamp>.json
```

安全约束：脚本不会写入或覆盖正式 `labels/`。重复运行默认也不会覆盖已有 `labels_auto/`；确认确实需要重新生成时才加 `--overwrite-auto`。

阈值选择建议：

- `0.25～0.35`：召回优先，框会更多，适合第一轮人工清洗；
- `0.45～0.60`：精度优先，但必须特别检查“自动判为无火”的图片，防止小火漏标；
- 不要把教师输出直接当真值。旧模型在新相机、曝光、场地和视角上会发生域偏移。

### 6.3 逐张人工复核并提升为正式标签

运行内置复核器：

```powershell
python scripts/review_pseudo_labels.py --source data/raw
```

默认按照风险排序：先显示自动判为负样本的图片，再显示低置信度图片。窗口按键：

| 按键 | 动作 | 是否进入正式 `labels/` |
|---|---|---|
| `A` | 接受当前自动框 | 是，复制自动标签 |
| `N` | 人工确认无火 | 是，写入空标签 |
| `D` | 丢弃该图，不用于训练 | 否 |
| `S` | 暂时跳过 | 否 |
| `B` | 返回上一张 | 不修改 |
| `Q`/`Esc` | 保存进度并退出 | 不修改当前图 |

复核进度保存在各场景的 `review_state.json`。正式 `labels/` 中已有文件默认受保护；只有明确使用 `--overwrite-final` 才允许覆盖。查看统计但不打开窗口：

```powershell
python scripts/review_pseudo_labels.py --source data/raw --list-only
```

`A/N` 适合快速筛选；框偏移、漏框或多框粘连的样本，应在接受后用 CVAT、Label Studio、LabelImg 等工具打开 `images/` 和正式 `labels/` 做精修。完成后建议人工抽查全部测试场景和至少 10% 的训练场景。

### 6.4 正式标签约定

本项目默认只训练火焰检测，因此类别必须只有：

```text
0 fire
```

标注约定建议固定如下：

- 框住当前可见火焰的外轮廓，不把大块烟雾或灯光一起框入；
- 分离且不相连的火焰分别标框；同一连续火焰用一个框；
- 只剩极少像素、肉眼无法确定是否是火的区域不要强行标；
- 屏幕/海报里的火焰是否作为真火，要按部署目标决定。消防告警通常应标为负样本，并保持全数据一致；
- 无火图片必须有一个同名的空 `.txt`，否则准备脚本会认为它“尚未标注”并跳过。

YOLO 每行标签格式是：

```text
class_id x_center y_center width height
```

后四项都按图像宽高归一化到 `[0, 1]`。把每张图片导出的 txt 放到对应场景的 `labels/`，文件名与 `images/` 中图片一致。例如：

```text
images/20261020_120000_000001_000001.jpg
labels/20261020_120000_000001_000001.txt
```

建议先让一人写出 20～50 张标注规范示例，再批量标注；完成后抽查至少 10%。框标准不一致通常比少量框偏差更伤模型。

如果确实要把火焰和烟雾训练成两个类别，自动标注时改成：

```powershell
python scripts/pseudo_label_yolov5.py --source data/raw `
  --python C:\Users\31866\miniconda3\envs\yolov5\python.exe `
  --class-map 0:0 1:1
```

后续生成数据集时必须同步使用 `--class-names fire smoke`。不要在同一批实验中途改变类别定义。

## 7. 按场景生成训练集并检查

以下命令以整个场景为最小单位做 70%/20%/10% 拆分，相邻帧不会被随机分到训练和验证集：

```powershell
python scripts/prepare_dataset.py --source data/raw --output data/fire `
  --train 0.7 --val 0.2 --test 0.1 --seed 42 --class-names fire
python scripts/check_dataset.py --data data/fire/dataset.yaml
```

`prepare_dataset.py` 为防误覆盖，要求输出目录为空。重新生成时请先确认旧数据无用，再手工清空 `data/fire/`（保留 `.gitkeep` 也可以改用一个新的输出目录）。它会生成：

- `data/fire/dataset.yaml`：训练入口，`path` 是本机绝对路径；
- `data/fire/split_manifest.json`：场景归属、样本数量和被跳过的未标注图；
- `images/{train,val,test}` 与 `labels/{train,val,test}`。

准备脚本只读取每个场景的正式 `labels/`，不会读取 `labels_auto/`。因此未复核、暂跳过或丢弃的教师预测不会悄悄进入训练集。若训练 `fire/smoke` 两类，改用 `--class-names fire smoke`，并确保正式标签只含类别 ID 0 和 1。

若数据较大，可加 `--link` 尝试使用硬链接节省磁盘；原始数据和输出必须在同一 NTFS 卷。

至少准备 3 个场景才可能让 train/val/test 各有一个场景；真实评估建议远多于这个数量。确认 `check_dataset.py` 输出“检查通过”后再训练。

## 8. 训练 YOLO26

### 8.1 第一轮完整实验

在至少完成若干独立场景的采集和人工复核后，第一轮实验按下面顺序执行：

```powershell
# 1. 按场景构建数据，labels_auto 不会被读取
python scripts/prepare_dataset.py --source data/raw --output data/fire_exp01 `
  --train 0.7 --val 0.2 --test 0.1 --seed 42 --class-names fire

# 2. 在训练前阻断缺标签、非法框和类别越界
python scripts/check_dataset.py --data data/fire_exp01/dataset.yaml

# 3. 两轮冒烟训练，只验证代码、CUDA、数据和保存流程
python scripts/train.py --data data/fire_exp01/dataset.yaml `
  --model yolo26n.pt --epochs 2 --batch 8 --name exp01_smoke_test

# 4. 初步 YOLO26s 基线
python scripts/train.py --data data/fire_exp01/dataset.yaml `
  --model yolo26s.pt --epochs 100 --imgsz 640 --batch 8 --device 0 `
  --name exp01_yolo26s_rgb

# 5. 训练和阈值选择结束后，只在冻结测试集上做正式评估
python scripts/evaluate.py `
  --model runs/train/exp01_yolo26s_rgb/weights/best.pt `
  --data data/fire_exp01/dataset.yaml --split test --imgsz 640 --batch 8 `
  --name exp01_yolo26s_rgb_test
```

评估脚本会在 `runs/eval/<name>/evaluation_summary.json` 保存指标和逐图速度。数据尚未采集或正式 `labels/` 为空时不能开始训练；当前仓库不会拿旧 YOLOv5 数据冒充新的 RGB-D 场景实验。

建议把第一轮定义为可追溯实验：固定 `seed=42`、场景划分、教师权重哈希、自动标注阈值和人工复核状态。第二轮只能改变一项主要变量，例如增加困难负样本、提高输入尺寸或更换模型规模。

### 8.2 两轮启动策略

先用小模型和 2 个 epoch 验证数据、CUDA、保存路径全部正常：

```powershell
python scripts/train.py --model yolo26n.pt --epochs 2 --batch 8 --name smoke_test
```

然后在 RTX 3060 12GB 上开始基线训练。推荐先用 `yolo26s.pt`、640 输入和 batch 8：

```powershell
python scripts/train.py --model yolo26s.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --name fire_yolo26s_baseline
```

训练产物位于：

```text
runs/train/fire_yolo26s_baseline/
├─ weights/best.pt
├─ weights/last.pt
├─ results.csv
├─ results.png
├─ confusion_matrix.png
└─ PR_curve.png
```

意外中断后完整续训：

```powershell
python scripts/train.py --resume runs/train/fire_yolo26s_baseline/weights/last.pt
```

### 8.3 RTX 3060 12GB 参数起点

| 参数 | 建议起点 | 什么时候调整 |
|---|---:|---|
| model | `yolo26s.pt` | 更看重速度用 `n`；数据足够且速度允许再试 `m` |
| imgsz | 640 | 远处小火漏检时试 768/960，同时下调 batch |
| batch | 8 | OOM 改 4/2；`-1` 让框架自动估计 |
| epochs | 100 | 曲线仍在改善可到 150～200；已过拟合就提前停 |
| patience | 30 | 小数据波动大可增至 50 |
| freeze | 0 | 极小数据集可先试 10，但最终通常应全量微调 |
| lr0 | 不指定 | 先相信权重内推荐配置；只有曲线明显震荡/不收敛再改 |
| mosaic | 0.7 | 小目标不足可稍增；形状不自然或误报增多则降低 |

显存不足时按这个顺序处理：`batch 8 → 4 → 2`，然后 `imgsz 640 → 512`，最后才换 `yolo26s → yolo26n`。不要同时改变多个变量，否则无法判断哪项有效。

远处小火是主要目标时可试：

```powershell
python scripts/train.py --model yolo26s.pt --epochs 150 --imgsz 960 --batch 2 --name fire_yolo26s_960
```

如果训练集能完全拟合但验证集差，优先增加不同场景和困难负样本，不要先堆 epoch。如果训练/验证都差，先检查漏标、框规范、目标尺寸和类别定义，再考虑增大模型或分辨率。

### 8.4 怎么看结果和选择阈值

- `mAP50-95`：主要综合指标，越高越好；
- Recall：真实火焰被找回的比例。消防场景通常比单纯追求 Precision 更重视 Recall；
- Precision：告警中有多少是真的，反映误报；
- PR 曲线：用独立验证集选择部署 `conf`，不要在测试集上反复调；
- confusion matrix 和验证预测图：定位漏标、灯光误报、小火漏检等具体模式。

最终测试集只用于冻结模型和阈值后的单次报告。除 mAP 外，还应统计“每小时无火视频误报次数”和“从起火到稳定告警的延迟”。这两个指标更接近实际消防需求。

## 9. 推理与部署

### 9.1 先对图片或录像做二维回放

```powershell
python scripts/infer_media.py --model runs/train/fire_yolo26s_baseline/weights/best.pt --source path\to\test.mp4 --conf 0.40
```

结果默认写入 `runs/predict/fire/`。也可把 `--source` 设置为图片、目录或普通摄像头编号 `0`。

### 9.2 Gemini 336L 实时检测和定位

```powershell
python scripts/infer_rgbd.py --model runs/train/fire_yolo26s_baseline/weights/best.pt --conf 0.40 --min-consecutive 3
```

界面左侧为 RGB 检测结果，右侧为对齐深度。按 `Q` 退出。事件逐帧追加到 `runs/infer/events.jsonl`。可同时录像：

```powershell
python scripts/infer_rgbd.py --model runs/train/fire_yolo26s_baseline/weights/best.pt --conf 0.40 --record runs/infer/demo.mp4
```

常用部署参数：

- `--conf 0.25～0.60`：越低召回高、误报多；应由验证集和无火长视频确定；
- `--min-consecutive 3`：连续 3 帧检测到才将 `alarm_active` 置真，抑制单帧闪烁；正式系统还应加入时间窗口、告警保持和人工复核；
- `--depth-roi base`：默认取框底部，`center`/`box` 可用于对比；
- `--min-depth`、`--max-depth`：按现场有效范围限制深度；
- `--software-align`：硬件对齐异常时切换；
- `--no-show`：无人值守运行时关闭窗口。

单条 JSONL 事件结构类似：

```json
{
  "captured_at_utc": "2026-10-20T12:00:00+00:00",
  "alarm_active": true,
  "detections": [
    {
      "class_name": "fire",
      "confidence": 0.91,
      "bbox_xyxy": [120, 80, 260, 310],
      "depth_is_proxy": true,
      "location": {
        "depth_mm": 1840.0,
        "xyz_m": [-0.21, 0.14, 1.84],
        "valid_ratio": 0.83
      }
    }
  ]
}
```

`location: null` 表示采样区域没有足够有效深度，代码不会用 0 或随机背景值伪造位置。

### 9.3 TensorRT FP16 加速

PyTorch `best.pt` 稳定后再导出。RTX 3060 推荐先用固定 640、batch 1、FP16：

```powershell
python scripts/export_model.py --model runs/train/fire_yolo26s_baseline/weights/best.pt --format engine --imgsz 640 --quantize 16 --device 0
```

导出的 `.engine` 通常在权重旁边，可直接交给同一个推理脚本：

```powershell
python scripts/infer_rgbd.py --model runs/train/fire_yolo26s_baseline/weights/best.engine --conf 0.40
```

需要跨框架时导出 ONNX：

```powershell
python scripts/export_model.py --model runs/train/fire_yolo26s_baseline/weights/best.pt --format onnx --imgsz 640 --quantize 16
```

说明：

- TensorRT engine 与 GPU 架构、TensorRT/驱动环境强相关，应在最终部署机上构建；迁移机器或大幅升级环境后重新导出；
- 首次 TensorRT 导出可能需要下载/安装额外依赖并耗时数分钟；
- 先比较 `best.pt` 与 `.engine` 在同一验证集上的精度，再换生产模型；
- INT8 需要代表性校准集，精度风险高于 FP16。确实要试时使用 `--quantize 8 --data data/fire/dataset.yaml`；
- `--nms end-to-end` 使用 YOLO26 一对一无 NMS 头，速度可能更好，但必须重新验证精度；默认保持框架常规输出最稳妥。

## 10. 深度定位校验

正式部署前，在 0.5 m、1 m、2 m、3 m 等已知距离放置不透明标定板：

1. 运行采集器，确认 RGB 边缘和深度边缘重合；
2. 用板上的目标框测试 `base`、`center` ROI 的 Z 偏差；
3. 分别在画面中心和四角测量，记录平均误差和 95% 误差；
4. 再对安全火源测试“燃烧物距离”与输出 Z 的关系；
5. 若火焰框内无深度，保留 `location=null`，不要用上一次距离冒充当前距离。

若需要把相机 XYZ 变成房间/机器人坐标，还需单独标定相机到世界坐标系的刚体变换 `P_world = R * P_camera + t`。本项目当前只输出相机坐标，不假设相机安装位姿。

## 11. 常见问题

### `torch.cuda.is_available()` 是 False

通常是装了 CPU wheel、NVIDIA 驱动过旧或虚拟环境混用。激活 `.venv` 后重新按 4.2 安装 CUDA wheel，再检查 `where.exe python` 和 `python -m pip show torch`。

### 找不到 Orbbec 或拿不到帧

换主板 USB 3.x 口和合格数据线，关闭占用相机的 Orbbec Viewer，重新运行 `setup_env.py`，拔插设备。仍失败时用官方 Viewer 确认固件/流配置是否正常。

### RGB 与深度错位

先确认输出状态是 `hardware-d2c`；分别测试默认模式和 `--software-align`。同时检查彩色/深度图尺寸及 `meta/*.json` 的内参。不要对未经 D2C 的原始深度直接按 RGB 框索引。

### 训练报 CUDA out of memory

把 `--batch 8` 改为 4 或 2；仍不足再降低 `--imgsz` 或换 `yolo26n.pt`。结束其他占显存程序后重启训练。

### 灯光、夕阳、反光频繁误报

采集这些具体现场的困难负样本并创建空标签，加入下一轮训练。仅提高 `conf` 会同时漏掉弱小火焰，通常不能替代数据修正。

### 小火漏检

增加近似距离和背景的小火样本，检查是否漏标；再比较 `imgsz=768/960`。若火焰在原图仅几像素，再大模型也很难恢复信息，应调整相机视场或增加相机。

## 12. 推荐的迭代顺序

1. 先采 10～20 个独立场景，用 YOLOv5 教师生成 `labels_auto/`；
2. 逐张复核，尤其检查自动负样本、远距离小火和强反光，正式结果进入 `labels/`；
3. 按场景拆分数据并完成 `yolo26n` 两轮冒烟训练；
4. 用 `yolo26s` 得到 RGB 基线，并固定一个完全独立的测试集；
5. 在目标现场连续录制无火视频，整理误报片段为困难负样本；
6. 补采漏检的小火、遮挡、远距离样本，只改变一项训练变量做对比；
7. 冻结模型与阈值后做长时间稳定性、告警延迟和深度误差测试；
8. 最后再做 TensorRT、服务自启动、日志轮转、告警接口和系统级冗余。

这个代码是检测和定位原型，不应作为唯一消防报警链路。生产系统应与合规烟感/温感/火灾报警控制器、人工复核、故障自检和断网降级策略结合。

## 13. 测试与参考

不连接相机也可以检查核心深度数学和场景拆分：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall src scripts
```

上游资料：

- [Ultralytics YOLO26 模型文档](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics YOLO26 训练配方](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
- [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)
- [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/)
- [Orbbec pyorbbecsdk v2](https://github.com/orbbec/pyorbbecsdk)
- [Orbbec 彩色/深度对齐示例](https://github.com/orbbec/pyorbbecsdk/blob/v2-main/examples/beginner/03_color_and_depth_aligned.py)

Ultralytics 当前采用 AGPL-3.0，并提供企业许可选项。若系统将闭源商用或作为网络服务交付，请在发布前确认许可义务。Orbbec Python SDK 使用 Apache-2.0。

