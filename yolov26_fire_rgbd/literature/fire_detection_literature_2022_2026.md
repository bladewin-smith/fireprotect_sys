# 近年视觉火焰、火灾与早期烟雾检测论文调研（2022–2026）

> 检索日期：2026-10-22  
> 面向项目：YOLO26 + Orbbec Gemini 336L RGB-D 火焰识别与三维定位  
> 配套索引：`fire_detection_papers_2022_2026.csv`（截至 2026-10-21；本次新增条目以本文为准）

## 1. 检索口径与结论先行

本报告关注使用图像、视频、红外/热成像或多模态数据完成火焰/火灾/早期烟雾的分类、目标检测、定位和语义分割。时间范围以 2022–2026 年为主，优先收录：

1. NeurIPS、AAAI、WACV 等国际会议正式录用论文；
2. IEEE Transactions、Neurocomputing、Knowledge-Based Systems、Expert Systems with Applications、International Journal of Applied Earth Observation and Geoinformation、Combustion and Flame、Fire Technology 等高影响力或领域代表性期刊；
3. 与本项目直接相关的轻量化检测、多模态融合、距离估计、数据集和误报抑制工作。

论文信息优先核对官方会议论文集、IEEE Xplore、ScienceDirect、Springer 和期刊 DOI 页面。预印本只有在可补充开放全文时才列为辅助链接，不把仅有 arXiv 的工作标成正式顶会/期刊论文。

需要特别注意：

- 在 CVPR/ICCV/ECCV 主会中，近年直接研究“普通摄像机火焰框检测”的论文并不多。更集中的成果出现在 AAAI、NeurIPS、WACV，以及工业信息、遥感、视频分析和消防安全期刊。
- 截至本次检索，尚未发现把消费级主动 RGB-D 深度图作为检测网络输入、并对其火焰识别增益进行系统公开评估的代表性顶会/期刊工作；但 2026 年 FireDet 已使用双目图像、火焰分割和立体匹配完成小火焰三维定位，是与本项目最接近的新成果。它仍属于“RGB 检测/分割后做双目定位”，并不等于 RGB-D 四通道端到端检测。多模态火灾论文更常使用 RGB + 热红外（TIR/LWIR）；几何深度与温度/辐射信息不可等同。
- 与当前项目最稳妥的技术路线仍是：RGB 检测负责识别火焰，配准深度负责距离和 XYZ 定位；同时把 4 通道 RGB-D 早期融合作为消融实验，而不是预设它一定优于 RGB。

本次清单的范围覆盖如下。标为“相邻证据”的论文不直接检测火焰，但对烟雾干扰、传感器可靠性或消防场景鲁棒性有明确参考价值。

| 检索范围 | 文中覆盖 |
|---|---|
| AAAI | 2022 烟雾分割、2024 早期烟雾分割、2026 烟雾下三维传感鲁棒性 |
| NeurIPS | 2024 噪声标签烟雾分割；2025 SmokeViz、DetectiumFire、Fire360 数据与基准 |
| WACV | 2025 UAV 实时分割；2026 SmokeBench、SmokeCPC、MultiNatSmoke |
| 工业信息/电子/交通 Transactions | TII、TIE、TITS 中的边缘检测、多注意力监控和距离估计 |
| 遥感 | TGRS、Remote Sensing of Environment、IJAEOG、International Journal of Digital Earth |
| 视频分析 | TCSVT、Expert Systems with Applications、Neural Computing and Applications、Fire Safety Journal 视频方法 |
| 消防安全 | Fire Safety Journal、Fire Technology，以及面向工业燃烧的 Combustion and Flame |

## 2. 顶级/重要国际会议论文

### C1. CoSW: Conditional Sample Weighting for Smoke Segmentation with Label Noise

- 会议：NeurIPS 2024，主会
- 作者：Lujian Yao, Haitao Zhao, Zhongze Wang, Kaijie Zhao, Jingchao Peng
- 任务：烟雾语义分割、噪声标签鲁棒学习
- 核心方法：针对烟雾边缘透明、标注不一致的问题，使用多原型和条件样本加权；通过正则化原型内熵动态降低可疑像素标签的权重。
- 主要价值：说明烟雾/火焰这种软边界目标不能只依赖普通硬标签。对本项目而言，可用于设计标签质检、软标签或置信度加权策略；它不是目标检测器，不能直接替代 YOLO。
- 官方论文页：https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1366167e310460f2386c36152a897d6-Abstract.html
- DOI：https://doi.org/10.52202/079017-3389

### C2. FoSp: Focus and Separation Network for Early Smoke Segmentation

- 会议：AAAI 2024，主会 Technical Track
- 作者：Lujian Yao, Haitao Zhao, Jingchao Peng, Zhongze Wang, Kaijie Zhao
- 任务：小尺度、低对比、半透明早期烟雾分割
- 核心方法：Focus 模块聚合多分辨率信息以降低漏检；Separation 模块显式分离烟雾前景和无烟背景；Domain Fusion 模块平衡召回率与精度。
- 数据与结果：发布 SmokeSeg，包含 6,144 张真实烟雾像素级标注图；论文报告在 SYN70K、SMOKE5K、SmokeSeg 上取得当时最佳结果。
- 项目启示：可借鉴小目标特征融合和“难负样本/低对比前景”设计，尤其适合远距离小火苗和初期烟雾。
- 官方论文页：https://ojs.aaai.org/index.php/AAAI/article/view/28484
- 代码：https://github.com/LujianYao/FoSp
- DOI：https://doi.org/10.1609/aaai.v38i7.28484

### C3. Transmission-Guided Bayesian Generative Model for Smoke Segmentation

- 会议：AAAI 2022，主会 Technical Track
- 作者：Siyuan Yan, Jing Zhang, Nick Barnes
- 任务：烟雾分割与不确定性估计
- 核心方法：贝叶斯生成模型同时建模模型不确定性和标签不确定性，并以物理去雾中的透射特征约束低对比、半透明烟雾区域。
- 数据与结果：发布 SMOKE5K，包括约 1,400 张真实图和 4,000 张合成图及像素级标注；输出分割结果的同时提供不确定性图。
- 项目启示：安全告警系统不应只输出框和置信度；可增加不确定性/连续帧确认，避免单帧误报触发报警。
- 官方论文页：https://ojs.aaai.org/index.php/AAAI/article/view/20207
- 代码与数据：https://github.com/redlessme/Transmission-BVM
- DOI：https://doi.org/10.1609/aaai.v36i3.20207

### C4. Detecting Wildfires on UAVs with Real-Time Segmentation Trained by Larger Teacher Models

- 会议：WACV 2025，主会
- 作者：Julius Pesonen 等
- 任务：机载实时野火烟雾分割
- 核心方法：用 SAM 或框监督大型教师模型产生伪掩膜，再蒸馏到轻量 PIDNet 学生模型；只需要较便宜的框标注即可获得分割监督。
- 结果：论文报告 63.3% mIoU；在 Jetson Orin NX 上纯推理约 25 FPS，包含视频记录的完整流程约 6 FPS；实测可在 1.4–9.7 km 距离发现烟雾。
- 项目启示：已有 YOLOv5 模型可作为自动预标注/教师模型，再训练 YOLO26 或轻量分割网络。这比强行把 YOLOv5 权重迁移到 YOLO26 更合理。
- 官方论文：https://openaccess.thecvf.com/content/WACV2025/html/Pesonen_Detecting_Wildfires_on_UAVs_with_Real-Time_Segmentation_Trained_by_Larger_Teacher_Models_WACV_2025_paper.html
- 代码：https://gitlab.com/fgi_nls/public/wildfire-real-time-segmentation

### C5. SmokeBench: Evaluating Multimodal Large Language Models for Wildfire Smoke Detection

- 会议：WACV 2026，主会
- 作者：Tianye Qi, Weihao Li, Nick Barnes
- 任务：野火烟雾分类、粗粒度定位和检测基准
- 核心方法：构建四类任务，对 Idefics2、Qwen2.5-VL、InternVL3、Unified-IO 2、Grounding DINO、GPT-4o、Gemini 2.5 Pro 等模型进行统一评测。
- 结论：多模态大模型在烟雾面积较大时分类尚可，但早期小烟雾的精确定位仍明显不足。
- 项目启示：通用视觉语言模型目前不适合直接取代安全关键的专用检测器；可用于辅助数据审核、场景描述和困难样本发现。
- 官方论文页：https://openaccess.thecvf.com/content/WACV2026/html/Qi_SmokeBench_Evaluating_Multimodal_Large_Language_Models_for_Wildfire_Smoke_Detection_WACV_2026_paper.html

### C6. False Alarm Rectification for Early Smoke Segmentation

- 会议：WACV 2026，主会
- 作者：Hongjin Zhao, Weihao Li, Ge-Peng Ji, Nick Barnes
- 任务：早期烟雾分割与误报抑制
- 核心方法：SmokeCPC 使用图像级分类置信度作为先验，结合跨尺度融合和监督对比损失，专门抑制云、雾、水汽等类似背景造成的假阳性。
- 结果：论文在 SmokeSeg 上报告 61.83% IoU 和 0.28% FPR。
- 项目启示：火焰项目也应建立“高相似非火焰”负样本集，如夕阳、灯光、反光、焊花、红色衣物和屏幕视频，并把误报率作为独立指标。
- 官方论文：https://openaccess.thecvf.com/content/WACV2026/papers/Zhao_False_Alarm_Rectification_for_Early_Smoke_Segmentation_WACV_2026_paper.pdf

### C7. AusSmoke Meets MultiNatSmoke: A Fully-Labelled Diverse Smoke Segmentation Dataset

- 会议：WACV 2026，主会
- 作者：Weihao Li, Hongjin Zhao, Gao Zhu, Ge-Peng Ji, Nicholas Wilson, Marta Yebra, Nick Barnes
- 任务：跨地区真实野火烟雾分割数据集与泛化基准
- 数据：AusSmoke 含 15,248 张澳大利亚真实图像；MultiNatSmoke 汇总多国数据并提供 70,818 张真实烟雾图像及完整分割标注。
- 主要价值：论文专门评估跨数据集零样本泛化，显示地理与场景多样性比单一地区扩大样本量更有价值。
- 项目启示：采集 Gemini 数据时应按场所、光照、火源、距离和相机姿态建立元数据；测试集应包含训练阶段未出现的场景，而不能只随机拆分相邻视频帧。
- 官方论文页：https://openaccess.thecvf.com/content/WACV2026/html/Li_AusSmoke_meets_MultiNatSmoke_a_fully-labelled_diverse_smoke_segmentation_dataset_WACV_2026_paper.html

### C8. SmokeViz: A Large-Scale Satellite Dataset for Wildfire Smoke Detection and Segmentation

- 会议：NeurIPS 2025，Datasets and Benchmarks Track
- 作者：Rey Koki, Michael McCabe, Dhruv Kedar, Josh Myers-Dean, Annabel Wade, Jebb Stewart, Christina Kumler-Bonfanti, Jed Brown
- 任务：卫星烟羽检测/分割数据集与粗粒度标注对齐
- 核心方法：从 NOAA 烟雾分析员的时空区间标注出发，提出伪标签降维（PLDR），由父模型选出时间窗口中与烟羽最匹配的单幅卫星图像，从而解决标注与输入在时间、空间分辨率上的错配。
- 数据规模：精炼后包含超过 160,000 个人工标注。
- 项目启示：伪标签不只是扩大样本量，也可以用来“选择最可信帧、修正标签对应关系”。使用 YOLOv5 辅助标注 RGB-D 视频时，应优先保留稳定帧，并抽样复核低置信度、边界漂移和跨帧跳变。
- 官方论文页：https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b2116b636f64553d7fe33d87a701b06-Abstract-Datasets_and_Benchmarks_Track.html
- 数据：https://noaa-gsl-experimental-pds.s3.amazonaws.com/index.html#SmokeViz/
- DOI：https://doi.org/10.52202/085713-1383

### C9. DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding

- 会议：NeurIPS 2025，Datasets and Benchmarks Track
- 作者：Zixuan Liu, Siavash H. Khajavi, Guangkai Jiang
- 任务：火灾目标检测、视频理解、图像生成与视觉语言推理
- 数据：22,500 张高分辨率火灾相关图像和 2,500 个真实视频，覆盖不同火灾类型、环境与风险等级；同时提供目标框和场景文本描述。
- 主要价值：论文在目标检测、扩散生成和视觉语言推理上验证了数据用途。对本项目最直接的用途是作为外部 RGB 预训练/混合训练数据，以及补充少见场景和困难负样本；文本标注不能直接代替检测框质检。
- 官方论文页：https://proceedings.neurips.cc/paper_files/paper/2025/hash/cfff7d95e129ec7bfde9308726736834-Abstract-Datasets_and_Benchmarks_Track.html
- 项目与数据说明：https://github.com/ZixuanLiu4869/DetectiumFire
- DOI：https://doi.org/10.52202/085713-4742

### C10. Fire360: A Benchmark for Robust Perception and Episodic Memory in Degraded 360° Firefighting Video

- 会议：NeurIPS 2025，Datasets and Benchmarks Track
- 作者：Aditi Tiwari, Farzaneh Masoud, Dac Nguyen, Jill Kraft, Heng Ji, Klara Nahrstedt
- 定位：消防场景鲁棒感知基准，属于与火焰检测相邻的成果，不是专门的火焰框检测器。
- 数据与任务：228 个专业消防训练的 360° 视频，标注动作片段、物体位置和图像退化信息；支持视觉问答、时序动作描述、物体定位、安全推理和受损物体检索。
- 主要结果：在受损物体检索任务中，人类专家达到 83.5%，现有多模态模型明显落后，说明烟雾、低照度、热扰动和结构形变会同时削弱视觉感知与推理。
- 项目启示：部署测试集需要主动加入烟雾遮挡、低照度、镜头污损和热扰动等退化条件，不能只测试清晰火焰画面。
- 官方论文页：https://proceedings.neurips.cc/paper_files/paper/2025/hash/2a62bf102605004f241f620d6f6aa980-Abstract-Datasets_and_Benchmarks_Track.html
- 数据：https://uofi.box.com/v/fire360dataset
- DOI：https://doi.org/10.52202/085713-0990

### C11. Physically-Based LiDAR Smoke Simulation for Robust 3D Object Detection

- 会议：AAAI 2026，Technical Track on Computer Vision
- 作者：Shijun Zheng, Yu Guo, Weiquan Liu, Yu Zang, Siqi Shen, Ming Cheng, Cheng Wang
- 定位：烟雾环境下的三维传感器鲁棒性研究，不是火焰/烟雾检测器。
- 核心方法：在 Unity 中进行三维流体烟雾模拟，并建立考虑衰减、散射和多径效应的 LiDAR 感知模型；通过距离图融合把模拟烟雾注入真实驾驶点云，并用实采 LiSmoke 数据验证。
- 项目启示：该工作给出了“烟雾会使主动三维传感数据稀疏、产生噪声和错误量测”的直接证据。LiDAR 与 Gemini 336L 的成像原理不同，不能把其数值结果直接迁移，但应据此增加深度有效率、异常值和烟雾浓度分组测试。
- 官方论文页：https://ojs.aaai.org/index.php/AAAI/article/view/38348
- DOI：https://doi.org/10.1609/aaai.v40i16.38348

## 3. IEEE Transactions 与高水平期刊论文

### J1. FlameFinder: Illuminating Obscured Fire Through Smoke With Attentive Deep Metric Learning

- 期刊：IEEE Transactions on Geoscience and Remote Sensing，2024
- 作者：Hossein Rajoli, Sahand Khoshdel, Fatemeh Afghah, Xiaolong Ma
- 任务：烟雾遮挡下的热红外火焰识别
- 模态：配对 RGB + 热红外用于训练，推理重点使用热红外特征
- 核心方法：注意力深度度量学习结合中心损失、三元组中心损失等，使热域中的火焰和非火焰热点更可分。
- 结果：论文报告相对基线在 FLAME2 和 FLAME3 的无遮挡火焰检测准确率分别提升 4.4% 和 7%，并改善遮挡场景类间分离。
- 项目启示：这是本项目研究多模态融合最值得精读的论文之一，但其“热红外”不是 Gemini 336L 的“几何深度”。
- IEEE/DOI：https://doi.org/10.1109/TGRS.2024.3440880
- 开放全文：https://arxiv.org/abs/2404.06653

### J2. Early Wildfire Detection and Distance Estimation Using Aerial Visible-Infrared Images

- 期刊：IEEE Transactions on Industrial Electronics，2024
- 作者：Linhan Qiao, Shun Li, Youmin Zhang, Jun Yan
- 任务：早期野火分割、距离估计和地理定位
- 模态：无人机可见光 + 红外 + 相机位姿/GPS
- 核心方法：Attention U-Net 分割烟雾和疑似火焰；ORB-SLAM 恢复相机位姿并三角测距；几何模型完成可见光与红外配准以降低漏检误检。
- 项目启示：与“YOLO 框 + 深度/位姿定位”最接近。可借鉴检测、配准、距离估计分模块评估的系统设计。
- IEEE 页面：https://ieeexplore.ieee.org/document/10520273/
- DOI：https://doi.org/10.1109/TIE.2024.3387089

### J3. Satellite Image-Based Surveillance and Early Wildfire Smoke Detection Using a Multiattention Interlaced Network

- 期刊：IEEE Transactions on Industrial Informatics，2025
- 作者：Shubhangi Chaturvedi 等
- 任务：复杂天气下的火灾/烟雾图像分类
- 核心方法：交错融合 Vision Transformer 与 CNN，并使用多注意力机制处理云、雾、风暴和雪等干扰。
- 结果：论文报告在两个数据集上分别降低约 30% 和 6% 的误报，能识别面积约占图像 2% 的小烟雾；模型约 0.7M 参数、0.2 GFLOPs。
- 局限：主要是图像级分类，不直接输出火焰框或深度位置。
- IEEE 页面：https://ieeexplore.ieee.org/document/10886910/
- DOI：https://doi.org/10.1109/TII.2025.3528549

### J4. EdgeFireSmoke: A Novel Lightweight CNN Model for Real-Time Video Fire–Smoke Detection

- 期刊：IEEE Transactions on Industrial Informatics，2022
- 作者：Jefferson Silva Almeida, Chenxi Huang, Fabricio Gonzalez Nogueira, Surbhi Bhatia, Victor Hugo C. de Albuquerque
- 任务：边缘端视频火焰/烟雾分类
- 核心方法：面向摄像机、无人机和 IoT 设备设计轻量 CNN，在本地完成实时告警。
- 结果：论文报告单帧约 30 ms、准确率 98.97%、F1 95.77%。
- 局限：以帧级分类为主，高准确率不能直接代表目标检测 mAP，也不能证明跨场景泛化。
- IEEE 页面：https://ieeexplore.ieee.org/document/9684982/
- 代码：https://github.com/jefferson2021ufc/EdgeFireSmoke
- DOI：https://doi.org/10.1109/TII.2021.3138752

### J5. Efficient Fire Segmentation for Internet-of-Things-Assisted Intelligent Transportation Systems

- 期刊：IEEE Transactions on Intelligent Transportation Systems，2023
- 作者：Khan Muhammad, Hayat Ullah, Salman Khan, Mohammad Hijji, Jaime Lloret
- 任务：交通/监控场景下的轻量火焰分割
- 核心方法：深度可分离卷积、逐点分组卷积和通道混洗，在边缘设备精度与复杂度之间折中，并在新建及公共数据集上验证。
- 项目启示：如果后续希望获得比框更精确的火焰区域，可以先用 YOLO 定位 ROI，再在 ROI 内运行轻量分割网络。
- IEEE 页面：https://ieeexplore.ieee.org/document/9894370/
- DOI：https://doi.org/10.1109/TITS.2022.3203868

### J6. SmokePose: End-to-End Smoke Keypoint Detection

- 期刊：IEEE Transactions on Circuits and Systems for Video Technology，2023
- 作者：Tao Jing, Ming Zeng, Qing-Hao Meng
- 任务：烟柱起点和终点关键点定位
- 核心方法：层次化 Transformer 编码器和纯 Transformer 解码器，通过热图端到端预测烟雾起点与扩散终点，并发布专用关键点数据集。
- 项目启示：比“是否有烟”进一步提供烟源方向信息；可启发对火焰根部、中心点或喷射方向进行结构化标注。
- IEEE 页面：https://ieeexplore.ieee.org/document/10075636/
- DOI：https://doi.org/10.1109/TCSVT.2023.3258527

### J7. A RGB-Thermal Based Adaptive Modality Learning Network for Day–Night Wildfire Identification

- 期刊：International Journal of Applied Earth Observation and Geoinformation，2023（卷期为 2024 前后出版周期）
- 作者：Xue Rui, Ziqiang Li, Xueyu Zhang, Ziyang Li, Weiguo Song
- 任务：昼夜野火语义分割/识别
- 模态：配对 RGB + 热红外
- 核心方法：分别学习模态专属特征与共享特征，并根据场景自适应融合，而不是简单拼接两路输入。
- 结果：论文报告相对次优 RGB-T 方法平均提升 6.41% IoU 和 3.39% F1。
- 项目启示：若未来增加热成像相机，双分支/自适应融合通常比单纯四通道拼接更有研究价值。
- 论文页：https://www.sciencedirect.com/science/article/pii/S1569843223003783
- DOI：https://doi.org/10.1016/j.jag.2023.103554

### J8. FireDM: A Weakly-Supervised Approach for Massive Generation of Multi-Scale and Multi-Scene Fire Segmentation Datasets

- 期刊：Knowledge-Based Systems，2024
- 作者：Hongtao Zheng, Meng Wang, Zilong Wang, Xinyan Huang
- 任务：弱监督火焰图像与分割掩膜生成
- 核心方法：结合 SDXL、Stable Diffusion 2.1、文本提示和 Fire-Decoder，从扩散模型的交叉注意力/多尺度特征中生成火焰掩膜。
- 结果：只需约 100 张图像及掩膜进行训练；使用生成数据训练的分割器在 IoU、准确率、F1、AP 上比公共小型数据集训练至少提升约 5%。
- 项目启示：适合扩充稀缺场景，但生成样本必须与真实测试集严格分离，并人工检查火焰形态、遮挡和背景合理性。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0950705124001825
- 代码与数据：https://github.com/ZhengHongtao2001/FireDM
- DOI：https://doi.org/10.1016/j.knosys.2024.111547

### J9. FireSeg: A Weakly Supervised Fire Segmentation Framework via Pre-Trained Latent Diffusion Models

- 期刊：Expert Systems with Applications，2025
- 作者：Wei Zhang, Hongtao Zheng, Weiran Li, Shuwen Pan, Yan Liu
- 任务：少标注条件下的火焰分割
- 核心方法：使用预训练潜空间扩散模型提取多层目标特征和交叉注意力，再由带动态位置查询和高分辨率交叉注意力的 Flame-Decoder 输出掩膜。
- 结果：论文在 BowFire、Corsican、FLAME 上相对次优方法分别提高 4.10%、3.83%、4.24%；只用 20% 弱监督数据时 IoU 提升 9.23%。
- 项目启示：可作为“少量像素标注 + 大量框标注”路线的参考，但计算和实现复杂度明显高于 YOLO 检测。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0957417425004749
- DOI：https://doi.org/10.1016/j.eswa.2025.126852

### J10. Enhancing Drone-Based Fire Detection with Flame-Specific Attention and Optimized Feature Fusion

- 期刊：International Journal of Applied Earth Observation and Geoinformation，2025
- 作者：Qiang Wang, Shiyu Guan, Shuchang Lyu, Guangliang Cheng
- 任务：无人机小/中尺度火焰目标检测
- 核心方法：以 YOLOv10 为基线，引入 Flame-Specific Attention、Focal Modulation 和多注意力动态检测头，针对火焰颜色、形状及复杂照明误报进行优化。
- 数据与结果：Comprehensive Fire Scene Dataset 共 3,905 张图；论文报告 mAP@0.5 为 87.7%，比 YOLOv10 基线精度提高 5.7%，推理速度 182 FPS。
- 项目启示：与 YOLO26 火焰框检测最直接相关，值得重点复现其困难负样本、小目标层和注意力消融实验；论文的速度必须结合其具体硬件和输入尺寸解读。
- 论文页：https://www.sciencedirect.com/science/article/pii/S1569843225003024
- DOI：https://doi.org/10.1016/j.jag.2025.104655

### J11. Automatic Flame Detection: Evaluation of Deep Learning Algorithms Using a Custom Thermal Image Dataset

- 期刊：Fire Technology，2025
- 作者：M. Hamed Mozaffari, Yuchuan Li, Yoon Ko, Sneha Rao
- 任务：室内真实火灾热红外目标检测基准
- 核心方法：在 NRC Canada 控制房间火灾实验的 881 张热红外图像上比较 33 种深度学习模型，综合评估准确性、速度、可靠性、泛化和计算代价。
- 结论：论文认为 YOLOX 是精度、资源和速度之间较合理的起点之一；这是一篇模型选择基准，不是提出单一新网络。
- 数据限制：数据需要依 NRC 规定申请，并非直接公开下载。
- 项目启示：强调模型评估不能只看 mAP，应同时报告速度、泛化和失败案例；热红外依旧不同于主动深度。
- 官方页面：https://link.springer.com/article/10.1007/s10694-025-01810-1
- DOI：https://doi.org/10.1007/s10694-025-01810-1

### J12. Flame Lift-Off Detector Based on Deep Learning Neural Networks

- 期刊：Combustion and Flame，2024
- 作者：Mohsen Gharib, Paul Tischer, Olaf Schulze, Martin Gräbner, Andreas Richter
- 任务：工业燃烧中火焰脱火检测与脱火长度估计
- 核心方法：卷积自编码器和回归神经网络处理高速火焰图像，替代计算较慢的传统图像分析流程。
- 结果：文中处理 10,000 张图像约需 3.5 秒，而传统图像处理约需 60 分钟。
- 局限：场景是受控燃烧装置的火焰稳定性诊断，不等同于复杂环境的通用火灾检测。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0010218023005898
- DOI：https://doi.org/10.1016/j.combustflame.2023.113215

### J13. Visual Fire Detection Using Deep Learning: A Survey

- 期刊：Neurocomputing，2024
- 任务：综述
- 内容：系统回顾约 130 篇视觉火灾检测论文，把任务划分为分类、定位和分割；整理 11 个常用图像/视频数据集及评价指标，并总结误报、数据不足和泛化问题。
- 项目启示：适合作为文献入口和论文 Related Work 的骨架，但具体数据集许可、链接和基准数值仍应回到原始论文核对。
- 论文页：https://www.sciencedirect.com/science/article/pii/S092523122400746X
- DOI：https://doi.org/10.1016/j.neucom.2024.127975

### J14. An Open Flame and Smoke Detection Dataset for Deep Learning in Remote Sensing Based Fire Detection

- 期刊：International Journal of Digital Earth，2024
- 作者：Ming Wang 等
- 任务：跨平台火焰/烟雾目标检测数据集
- 数据：FASDD 超过 120,000 个异构样本，包含地面摄像机、无人机和卫星子集；提供 VOC、COCO、YOLO 和 TDML 等标注格式。
- 结果：论文使用 Swin Transformer 技术验证，在 FASDD_CV、FASDD_UAV、FASDD_RS 上报告 84.9%、89.7%、74.0% 的 mAP。
- 项目启示：可用于 YOLO26 预训练/混合训练和困难样本补充，但最终验证集必须保留 Gemini 336L 实拍数据，避免只在公开数据上表现良好。
- 论文页：https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2347922
- 数据与说明：https://github.com/openrsgis/FASDD
- DOI：https://doi.org/10.1080/10095020.2024.2347922

### J15. FireDet: A Framework for Precise Detection and Localization of Small-Scale Fire

- 期刊：Fire Safety Journal，2026
- 作者：Jinyang Dai, Qiuyang Cao, Xiaoning Song, Bence Pan, Qixing Zhang
- 任务：早期小尺度火焰检测与双目三维定位
- 核心方法：以 YOLOv11 为基线构建 FSDNet，引入 CSCConvNeXtV2、P2 小目标检测层和 CBAM；再用 PSO 优化双目标定参数，以 FSDNet-OTSU 分割动态火焰，并使用 CREStereo 立体匹配完成定位。
- 数据与结果：综合检测集 12,116 张图，小火焰测试集 9,209 张图；另采集 839 对棋盘格、120 对火焰标定图和 1,498 对 10–120 m 双目火焰图。论文报告 mAP@0.5 相对基线提升 2.5 个百分点，在小火焰集提升 3.0 个百分点；0–120 m 范围内平均定位误差低于 2 m。
- 项目启示：这是目前与 Gemini 项目最接近的文献。应复现其“检测精度”和“空间定位误差”分开评估的做法；但它基于双目左右图立体匹配，而本项目可直接使用相机输出的配准深度，因此更适合做后融合对照，不构成四通道 RGB-D 训练必然有效的证据。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0379711226001050
- 数据仓库：https://github.com/yang1002-life/small-fire_datasets
- DOI：https://doi.org/10.1016/j.firesaf.2026.104737

### J16. Fire Vigilance Pocket: An Intelligent APP for Real-Time Fire Hazard Quantification

- 期刊：Fire Technology，2025
- 作者：Zilong Wang, Tianhang Zhang, Xinyan Huang
- 任务：从火焰发现扩展到火灾危险量化
- 核心流程：YOLOv5 检出火源，Swin-Unet 在 ROI 中分割火焰，再根据图像参考尺度估计火焰高度和面积，最后由预训练火灾量热模型估计热释放率（HRR）。
- 项目启示：适合作为部署阶段的级联架构参考，即“YOLO26 检测 → ROI 精分割 → 深度尺度换算 → 距离、面积或风险量化”。论文也明确指出现有尺寸测量依赖参考尺度，并把 LiDAR 或立体视觉列为自动尺度获取方向。
- 局限：测量精度受视角、完整入镜、遮挡和训练火灾规模影响，不能从框面积直接推断真实火焰面积或 HRR。
- 官方论文页：https://link.springer.com/article/10.1007/s10694-025-01738-6
- DOI：https://doi.org/10.1007/s10694-025-01738-6

### J17. High-Performance Real-Time Fire Detection and Forecasting Framework for Industrial Cables

- 期刊：Fire Safety Journal，2024
- 作者：Wanfeng Sun, Haibo Gao, Cheng Li
- 任务：工业电缆火焰检测、几何量提取和短期发展预测
- 核心方法：先以 HSV 增强火焰特征并用 YOLOv8 检测，再把连续帧得到的火焰高度、宽度和纵向位置输入 iTransformer 预测传播趋势。
- 结果：论文报告火焰高度、宽度和纵向位置检测的 MAPE 分别为 3.49%–10.64%、2.45%–8.89% 和 1.61%–9.31%；预测模块的误差随预测变量和实验设置变化。
- 项目启示：提供了“检测结果 → 时序几何特征 → 趋势预测”的工业管线。可先在 RGB-D 视频中记录每帧框、距离和尺寸，后续再评估是否需要时序预测，不应在基础检测尚未稳定时提前耦合复杂预测器。
- 局限：实验在受控燃烧室和特定电缆上完成，不能直接代表普通室内外场景的泛化性能。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0379711224001413
- DOI：https://doi.org/10.1016/j.firesaf.2024.104228

### J18. 3DVSD: An End-to-End 3D Convolutional Object Detection Network for Video Smoke Detection

- 期刊：Fire Safety Journal，2022
- 作者：Yinuo Huo, Qixing Zhang, Yongming Zhang, Jiping Zhu, Jinjun Wang
- 任务：连续视频中的烟雾识别与定位
- 核心方法：直接输入连续帧，以 3D 卷积提取动态特征，再与静态特征结合，在序列最后一帧输出烟雾位置；同时构建时序烟雾视频数据集。
- 结果：论文在其数据与划分上报告 99.54% 准确率、1.11% 误报率和 0.14% 漏检率，并满足其实时要求。
- 项目启示：火焰闪烁和烟雾扩散都包含单帧模型没有利用的时序信息。工程上可先保留 YOLO26 单帧检测器，再增加轻量的连续帧确认/轨迹过滤基线，而不必立即改造成 3D CNN。
- 复核提醒：上述指标来自专用数据集，不能与跨场景 YOLO mAP 或实际误报次数/小时直接比较。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0379711222001679
- DOI：https://doi.org/10.1016/j.firesaf.2022.103690

### J19. Fire and Smoke Detection from Videos: A Literature Review Under a Novel Taxonomy

- 期刊：Expert Systems with Applications，2024
- 作者：Diego Gragnaniello, Antonio Greco, Carlo Sansone, Bruno Vento
- 任务：视频火焰/烟雾检测综述与场景化评估分类
- 内容：复核 153 篇论文和 17 个公开数据集，以画面中的火灾尺寸及背景活动程度建立应用导向分类；又对 9 个数据集中的 536 个视频补充统一元数据标注。
- 项目启示：数据拆分和报告应同时覆盖“小/中/大火焰 × 低/高背景活动”，而不是只给总体 mAP。其公开表格可用于筛选合适的视频数据集和困难场景。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0957417424016506
- 配套标注：https://github.com/MiviaLab/FireDetectionSurvey2023/blob/main/MIVIA_FIRE_DETECTION_VIDEO_ANNOTATIONS.xlsx
- DOI：https://doi.org/10.1016/j.eswa.2024.124783

### J20. FLAME: Fire Detection in Videos Combining a Deep Neural Network with a Model-Based Motion Analysis

- 期刊：Neural Computing and Applications，2025
- 作者：Diego Gragnaniello, Antonio Greco, Carlo Sansone, Bruno Vento
- 任务：视频火焰/烟雾候选检测与误报抑制
- 核心方法：单帧轻量目标检测器先保证召回率；随后使用背景建模、候选位置关联和运动一致性/持续性分析，排除静态火焰相似物及运动模式不符的假目标。
- 结果：论文报告运动过滤使 Precision 和 F-score 分别提高 18 和 10 个百分点，并可在嵌入式系统实时运行。
- 项目启示：非常适合部署端借鉴。对灯光、反射、车灯、红色移动物等误报，可比较“仅 YOLO26”“连续 N 帧确认”“背景/运动过滤”三种策略的误报/小时与报警延迟。
- 官方论文页：https://link.springer.com/article/10.1007/s00521-024-10963-z
- 代码与数据：https://mivia.unisa.it/large-fire-dataset-with-negative-samples-lfdn/
- DOI：https://doi.org/10.1007/s00521-024-10963-z

### J21. Near-Real-Time Wildfire Detection with Himawari-8/9 Data Integrating Multi-Scale Spatial–Temporal Features

- 期刊：International Journal of Applied Earth Observation and Geoinformation，2025
- 作者：Lizhi Zhang, Qiang Zhang, Qianqian Yang, Linwei Yue, Jiang He, Xianyu Jin, Qiangqiang Yuan
- 任务：地球同步卫星 10 分钟序列中的早期活跃火点检测
- 核心方法：多核注意力卷积提取不同尺度的空间差异，长短期 Transformer 建模不同时间窗口，再融合为火点分类图；训练中还构造了基于火灾特性的虚拟数据。
- 结果：论文报告平均 Fire Accuracy 为 88.25%、平均 False Alarm Rate 为 20.82%，并优于其比较的 JAXA 产品和代表模型。
- 项目启示：虽然卫星火点与近景火焰框不是同一任务，但“多时间尺度”和“必须单独报告误报率”值得迁移到视频部署评估；较高的 FAR 也提醒不能只摘录准确率。
- 论文页：https://www.sciencedirect.com/science/article/pii/S1569843225000639
- 代码与部分数据：https://github.com/eagle-void/MSSTF
- DOI：https://doi.org/10.1016/j.jag.2025.104416

### J22. Toward an Adaptable Deep-Learning Model for Satellite-Based Wildfire Monitoring with Consideration of Environmental Conditions

- 期刊：Remote Sensing of Environment，2023
- 作者：Yoojin Kang, Taejun Sung, Jungho Im
- 任务：跨地区、昼夜和观测条件的卫星活跃火点检测
- 核心方法：双模块 CNN 分别处理亮温相关主变量和太阳/卫星天顶角、相对湿度、地表温度等环境辅助变量，再进行融合。
- 结果：论文报告双模块模型 Recall 0.84、Precision 0.97，并降低了五个研究区域间的召回率波动；相对湿度和卫星天顶角有助于减弱环境与观测角度造成的性能变化。
- 项目启示：数据集元数据不应只记录类别，还应记录距离、光照、室内外、视角和深度有效率；这些条件既可用于分组评估，也可作为后续自适应融合的辅助输入。
- 论文页：https://www.sciencedirect.com/science/article/abs/pii/S0034425723003656
- DOI：https://doi.org/10.1016/j.rse.2023.113814

### J23. Spatiotemporal Dynamics for Fire Type Detection: Differentiating Lithium-Ion Battery, Benign, and Conventional Fires

- 期刊：Fire Safety Journal，2026
- 任务：监控视频中的火灾类型分类，不负责输出火焰目标框
- 核心方法：利用火焰运动、烟雾演变和闪烁等时空线索，把视频分为锂离子电池火灾、传统 A–F 类火灾、良性火源和非火灾场景。
- 结果：在比较的网络中 SlowFast 表现最稳定，论文报告 Accuracy 86.5%、Precision 88.5%、Recall 86.5%、F1 87.0%。
- 项目启示：基础定位可靠后，可以在检测 ROI 的短视频片段上增加第二阶段风险分类；但当前阶段不应把该分类器与 YOLO26 定位精度混在同一指标中。
- 论文页：https://www.sciencedirect.com/science/article/pii/S0379711226001578
- DOI：https://doi.org/10.1016/j.firesaf.2026.104789

## 4. 与当前 YOLO26 + RGB-D 项目最相关的阅读顺序

建议按以下顺序精读，而不是按发表时间顺序：

1. **Visual Fire Detection Using Deep Learning: A Survey** 与 **Fire and Smoke Detection from Videos...**：建立任务、数据集、场景分类和评价指标全景。
2. **FireDet**：直接参考“小火焰检测 + 双目定位”，并据此拆分检测误差和三维定位误差。
3. **FASDD** 与 **DetectiumFire**：评估可转换为 YOLO 格式的外部数据，补充火焰、烟雾和困难负样本。
4. **Enhancing Drone-Based Fire Detection...**：参考 YOLO 小火焰检测、注意力和特征融合消融。
5. **Early Wildfire Detection and Distance Estimation...**：参考“识别 + 配准 + 距离/坐标估计”的系统拆分。
6. **FLAME** 与 **3DVSD**：建立视频时序确认、运动过滤和误报/小时基线。
7. **FlameFinder** 与 **RGB-T Adaptive Modality Learning**：理解真正的多模态训练、共享特征与模态专属特征，同时牢记热红外不等于深度。
8. **Automatic Flame Detection...** 与 **Fire Vigilance Pocket**：设计速度、泛化、失败案例以及火焰尺度/风险量化评价。
9. **CoSW / FoSp / SmokeCPC / MultiNatSmoke / SmokeViz**：用于标签质量、早期小目标、误报抑制和跨场景泛化。
10. **FireDM / FireSeg**：真实数据基线建立后，再考虑合成数据和弱监督分割。

## 5. 可以直接转化为项目实验的研究问题

### 5.1 三条基线必须同时保留

使用相同、按场景隔离的 train/val/test 划分比较：

1. YOLOv5 RGB（现有模型）；
2. YOLO26 RGB；
3. YOLO26 RGB 检测 + 深度后融合定位；
4. 可选：YOLO26 RGB-D 四通道早期融合。

只有第 4 项在同一测试集上稳定优于第 2/3 项，才能说明深度参与识别有实际收益。

### 5.2 指标不要只用 mAP

至少报告：

- mAP50、mAP50-95、Precision、Recall；
- 按场景计算的误报数/小时；
- 小目标 AP 或按框面积分组的 AP；
- 端到端延迟和 FPS，而不仅是网络前向时间；
- 有效深度比例、`location=null` 比例；
- 距离 MAE/RMSE、XYZ 误差；
- 白天/夜间、室内/室外、近/远距离、遮挡和强反光分组结果。

### 5.3 RGB-D 研究的关键风险

- 主动双目深度通常无法稳定测量透明/半透明火焰表面，框内深度可能属于燃烧物或背景。
- RGB-D 模型可能错误学习“某个距离/某个背景更容易有火”，造成换场景失效。
- 每张深度图单独 min-max 归一化会破坏物理距离一致性；应使用固定量程并保留无效深度掩码。
- RGB 与深度必须严格时间同步和 D2C 对齐，训练增强必须保持几何一致。
- 如果未来真正希望通过第二模态提升“火焰是否存在”的判断，热红外通常比几何深度更直接，但会增加硬件和标定成本。

### 5.4 根据新增论文优先补做的三个实验

1. **检测与定位分开评估**：参考 FireDet，先计算 YOLO26 框检测指标，再只对真阳性框计算距离/XYZ 误差；同时按距离区间、框面积和有效深度比例分组，避免检测失败被定位指标掩盖。
2. **单帧与时序策略对照**：参考 FLAME 和 3DVSD，比较单帧报警、连续 N 帧确认、轨迹/运动过滤三种策略，报告误报次数/小时、漏报率和首次报警延迟。
3. **伪标签质量闭环**：参考 WACV 教师模型工作和 SmokeViz，记录 YOLOv5 预标注置信度、人工修改比例、被删除框比例及跨帧跳变；优先人工复核低置信度、小目标、遮挡和深度无效样本，而不是把所有自动框直接作为真值。

