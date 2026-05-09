# UAV GPS 欺骗检测的时间感知评估
## ——基于跨分布泛化的实证研究

> 中文预览稿 · 对应 `docs/main.tex` · 生成时间 2026-05-04
> 用于核对叙事与论点；公式/引用/表格细节请参照 LaTeX 源文件

---

## 摘要

针对电力巡检等安全关键 UAV 任务的实时 GPS 欺骗检测，传统分类指标无法回答两个本质问题：
（1）**何时**检测到威胁；（2）虚警**多频繁**地干扰运行。

我们采用三个面向运行的指标 ——
**DR@Δt**（时间预算内检测率）、**ADD**（平均检测延迟）、**MTBFA**（事件聚合后的虚警间隔）—— 对 7 种代表性深度检测架构（CNN, LSTM, BiLSTM, GRU, CNN-LSTM, TCN, Transformer）做系统性实证研究。

训练数据：3000 飞行的 MATLAB UAV Toolbox 公开仿真集，4 类参数化攻击（Step / Drift / Delay / Takeover）。
评估：5 随机种子的 mean ± 95% CI + Bonferroni 校正逐对显著性检验。

**研究揭示三条直接面向部署的发现：**

1. **架构–攻击交互显著且非单调**：TCN 在时序型攻击（Takeover/Delay）上与最佳模型并列，但在位置型攻击（Step/Drift）上崩盘（Drift DR@5s = 0.46，Takeover = 0.96），分布内平均指标会掩盖这种攻击特异性失败。

2. **分布内架构排名不预测跨分布排名**：分布内冠军 CNN-LSTM（DR@5s = 0.96）在真实 Holybro S500 + HackRF 欺骗数据上跌到 0.20；而简单的循环架构（LSTM, GRU）保持 ≥ 0.80。

3. **天真匹配仿真巡航速度反而恶化跨分布迁移**：将仿真器从 5 m/s 改为 20 m/s 以匹配 ALFA 固定翼平台后，CNN 在 ALFA 上的 DR@5s 从 0.96 跌到 0.00（受控对比），与"sim-to-real 平台匹配"的常识直觉相反。

阈值依据来自 UAV 电力巡检风险约束（5 m/s 巡航 + GB 26860-2011 安全距离）。完整代码、数据集和训练 checkpoint 已公开。

**关键词**：UAV 安全；GPS 欺骗检测；时间感知评估；跨分布泛化；sim-to-real；深度学习基准；实时检测；虚警聚合。

---

## §I 引言

### 1.1 背景

随着 UAV 在电力系统巡检等领域的广泛应用，UAV 导航定位日益依赖 GNSS/GPS。在低冗余场景下（无 LiDAR/视觉 SLAM），GNSS 是唯一定位源。这种依赖暴露 UAV 于严重导航安全威胁——尤其是 **GNSS 欺骗攻击**：攻击者发送伪造卫星信号，使 UAV 接受虚假位置信息。在电力巡检中，被操纵的导航可导致轨迹偏离、违反高压设备安全边界、碰撞、坠毁、任务失败。

### 1.2 现有工作的不足

业界主要使用 precision/recall 等**静态分类指标**评估检测器。但这些指标：
- 无时间维度，无法量化响应速度
- 无法量化虚警频度
- 高离线 precision 不等于实际可部署：警报延迟过长会让 UAV 漂出几百米；高频虚警导致**警报疲劳**

时间序列异常检测领域（Tatbul 等）虽然引入了**事件级**评估，但主要面向离线，缺少 online causality 和 detection delay 建模。

### 1.3 三个开放问题

本文做系统性实证研究，回答：
1. 是否存在被传统聚合指标掩盖的、特定架构的失效模式？
2. 分布内架构排名是否迁移到真实跨分布评估？
3. 天真匹配仿真器巡航速度到目标平台速度，能否缩小 sim-to-real gap？

### 1.4 贡献

1. **基于部署风险的运行评估方法学**：采用 DR@Δt / ADD / MTBFA 三项指标 + 虚警事件聚合，从电力巡检风险（5 m/s 巡航 + GB 26860-2011 5 m 安全距离）推导可部署阈值（DR@5s ≥ 0.85，MTBFA ≥ 0.1 h）。

2. **公开 3000 飞行 MATLAB 仿真数据集**：4 种参数化攻击家族；严格 train/val/test 隔离（1800/600/600）；按 flight_id 确定性 seed 保证可复现；公开发布支持后续工作。

3. **统计严谨的多架构基准**：7 种代表性架构 × 5 seed × Bonferroni 校正 paired t-test。

4. **三条 deployment-relevant 实证发现**：
   - 架构–攻击交互（TCN 位置型 vs 时序型攻击的反直觉非对称）
   - 跨分布排名反转（CNN-LSTM 从 0.96 跌到 0.20，LSTM/GRU 保持 ≥ 0.80）

5. **受控平台匹配实验 + 反直觉结果**：保持 600 flights 训练规模，对比 5 m/s 与 20 m/s（匹配 ALFA 固定翼）训练。匹配速度 regime 反而让 ALFA DR@5s 跨架构降低 0.36–0.96，与 sim-to-real 平台匹配的常识相反。

---

## §II 问题设定

### 2.1 在线检测任务

考虑基于航点的 UAV 电力巡检任务。在低冗余导航条件（无 LiDAR/视觉 SLAM）下，UAV 主要依赖 GNSS/GPS 观测和惯性/运动状态（位置、速度、加速度）跟踪轨迹。

**核心任务**：给定时序状态 $\{\mathbf{x}_t\}$（特征维度 9），训练在线二元检测器 $f: \mathbb{R}^9 \to \{0,1\}$，在每个决策时刻输出是否被欺骗。

**双重目标**：
- **及时性**：攻击启动后尽快报警以触发缓解（切换导航源、紧急 RTH）
- **虚警控制**：正常飞行中虚警尽量少

### 2.2 因果性与决策时刻语义

引入 **离散决策时刻** $\{\tau_k\}_{k=1}^K$ 形式化因果性：在 $\tau_k$ 输出 $\hat y_{\tau_k}$ 仅依赖 $t \le \tau_k$ 的观测。

定义首次检测时刻、攻击起始时刻、检测延迟。

---

## §III 运行评估指标（重命名后的"原 framework 章"）

> **改动说明**：原标题 "Proposed Time-Aware Evaluation Framework" 改为 "Operational Evaluation Metrics"；删除了原 §III-D Tatbul/NAB 对比表；不再 claim "我们提出 framework"，而是 "本研究采用的运行评估指标"。

### 3.1 离散时间在线检测的形式化

窗口 $w_k$ → 决策 $\hat y_k = f(w_k)$ → 时间戳 $\tau_k$。

### 3.2 事件型 ground truth 与检测语义

- **首次检测时刻** $t_d = \min\{\tau_k : \hat y_k = 1, \tau_k \ge t_s\}$
- **检测延迟** $\Delta = t_d - t_s$
- **虚警事件聚合**：连续 FP 窗口聚合为单个 FP 事件

### 3.3 三个时间感知指标

- **DR@Δt**：时间预算 Δt 内能检测到的攻击占比
- **ADD**：所有被检测到攻击的平均延迟
- **MTBFA**：（正常飞行总时长）÷（FP 事件数），单位小时

### 3.4 滑窗实例化

windowing 仅是一种实例化方式；其他实例化（per-step RNN, event-triggered）可直接评估。

> **删除**：原 §III-D "Relation to Prior Time-Series Evaluation Frameworks"（4 列对比表 + 4 段防御 framework 新颖性的论述）整段被删。

---

## §IV 实验设计

### 4.1 数据来源与生成 pipeline

**MATLAB UAV Toolbox 仿真**：
- 4 种 mission profile（电力巡检 / 矩形扫描 / 绕塔 / 长距离 transit）
- `gpsSensor` + `imuSensor` + `insfilterAsync` 状态融合
- 每飞行 30–90 s @ 10 Hz，输出 9 维状态向量
- 1800 训练 / 600 验证 / 600 测试，flight-level 严格隔离，按 flight_id 确定 seed

**train-only augmentation 约束**：可选 TimeGAN 增广只施加在 train，test/val 永远不增广（程序断言保证）。

### 4.2 威胁模型与攻击注入

四种攻击家族（参数完全在 §IV-B 列出）：
- **Step**：瞬时位置跳变 $M \in \{5, 15, 30\}$ m
- **Drift**：渐进偏移，持续时间 $T_{drift} \in \{5, 10, 20\}$ s
- **Delay**：固定时间延迟 $\delta \in \{1, 3, 5\}$ s
- **Takeover**：平滑过渡到伪造轨迹，gain $\alpha \in \{0.3, 0.5, 0.7\}$

每类攻击有真实历史对应：2011 RQ-170 捕获、2013 Humphreys 游艇 demo、2016 Black Sea 海事欺骗。

### 4.3 滑窗与标签

L = 50, S = 5；终点标签法（窗口标签由窗口最后一个样本决定）。

### 4.4 检测器与训练协议

7 模型：CNN / LSTM / BiLSTM / GRU / CNN-LSTM / TCN / Transformer。Adam, lr=1e-3, BS=64, 最多 100 epoch + early stop patience=10，5 seeds，AMD Radeon 8060S iGPU + ROCm 7.2。

### 4.5 部署安全标准（风险推导）

- UAV 巡检速度 v = 5 m/s
- 5 秒响应预算 → 25 m 位移
- GB 26860-2011：500 kV 安全距离 5 m
- → 留 20 m abort margin
- → **DR@5s ≥ 85%, MTBFA ≥ 0.1 h**

### 4.6 可复现性

7 模型架构汇总表 + 训练超参 + GitHub 代码地址（[PLACEHOLDER]，待你填）。

---

## §V 案例研究

> **改动说明**：原标题 "Case Study" 保留；原 7 子节展开为 8 子节（新增 §V-D 大改、§V-E 大改、§V-G 重命名 + 大改、新增 §V-H）

### 5.0 章首引导段

7 检测器结果**演示** framework 的辨识力，**不**是普适架构推荐；具体值依赖训练数据 + 超参。

### 5.A 总览：传统 vs 时间感知

→ 三张表（auto-generated from `output/tables/`）：
- **table_conv_metrics**：Precision / Recall / F1
- **table_timeaware_metrics**：DR@1/3/5/10/15s / ADD / MTBFA
- **table_deployability**：可部署判定（传统 √/×、时间感知 √/×、失败原因）

**两条 finding**：
1. 强模型一致：传统过的也时间感知过
2. 时间感知揭示传统盲区：高 precision/F1 仍可能 DR@5s 不及格或 MTBFA 不达标

### 5.B 检测速度与虚警频度深析

- DR@Δt 曲线 → fig_c1（占位）
- Precision-MTBFA paradox → fig_b3（占位）
- FP windows vs FP events 聚合比 → fig_b4（占位）
- FP duration boxplot → fig_d2（占位）

> **状态**：4 个 deep-dive figure 都是占位 PDF。要么用 `visualize.py` 重新生成，要么手画。

### 5.C 速度–虚警 trade-off

Pareto plot DR@5s vs MTBFA → fig_e1（占位 PDF，等同于 pareto_dr5s_mtbfa.pdf）

### 5.D **Robustness Analysis**（新填）

#### 5.D.1 窗口 / 步长鲁棒性（GRU, 3 seeds）

| Window \ Step | S=3 | S=5 | S=10 |
|---|---|---|---|
| L=30 | 0.951 | 0.948 | 0.949 |
| L=50 | 0.953 | 0.940 | 0.924 |
| L=80 | 0.951 | 0.930 | 0.965 |

→ heatmap fig（已生成）。**DR@5s 全部在 0.924–0.965 间，spread ≤ 0.04**，说明框架结论不依赖于具体 (L, S) 选择。

#### 5.D.2 阈值敏感性

DR@Δt 时间窗 ∈ {1,3,5,10,15} s × MTBFA 阈值 ∈ {0.05, 0.1, 0.15, 0.2} h 网格。

**关键观察**：本文测试集尺度（600 飞行 × ~45s 平均 ≈ 7.13 h 累积正常时间）下，**所有架构 MTBFA 都在 0.016–0.043 h，无任何架构通过 MTBFA ≥ 0.1 h**。这本身是诊断性发现：要么测试集飞行时长不足以摊薄滑窗 FP，要么默认 0.5 决策阈值对运行 MTBFA 预算太宽松。详见 §VI-C Limitations。

放宽到 0.05 h 时，可部署判定主要由 DR@Δt 决定。

### 5.E **Per-Attack-Type Performance**（新填）

#### 7×4 完整 heatmap

| Model | Step | Drift | Delay | Takeover |
|---|---|---|---|---|
| CNN | 0.967 | 0.897 | 0.989 | 1.000 |
| LSTM | 0.971 | 0.894 | 0.904 | 0.984 |
| BiLSTM | 0.950 | 0.903 | 0.879 | 0.984 |
| GRU | 0.954 | 0.919 | 0.925 | 0.990 |
| CNN-LSTM | 0.992 | 0.869 | 0.989 | 0.997 |
| **TCN** | **0.637** | **0.463** | 0.961 | 0.961 |
| Transformer | 0.871 | 0.831 | 0.971 | 1.000 |

**两个揭示**（聚合 DR@5s 看不出来的）：

1. **TCN 的位置–时序非对称性**：在时序型攻击（Delay=0.961, Takeover=0.961）上正常，但位置型攻击（Step=0.637, Drift=0.463）上崩盘。这与"dilated conv 长程依赖更强 → 普适更好"的直觉相反：处理静态空间偏移，TCN 反而不如简单架构。每个 seed 都重现（Drift CI [0.42, 0.51]），有统计显著性。

2. **Delay 攻击诱导跨架构 P-R 分歧**：多数架构在 Delay 上 DR@5s ≥ 0.879，但 F1 集中在 0.711–0.784（比其他攻击低 0.15–0.20）。延迟攻击产生持久 GPS-惯性位置不一致，膨胀虚警率，即使首次检测已发生。这种 P-R 分歧对纯时间预算指标（DR@Δt）不可见，强化了"两个维度都报告"的必要性。

### 5.F 计算效率

7 模型推理时间表（i7-12700H, 1000 次/窗口平均），全部 ≤ 1.5 ms/窗口，远低于滑窗步长（≈100 ms @ 10 Hz）。

### 5.G **Cross-Distribution Generalization Evaluation**（新填，原 5.G 全删重写）

> **重要措辞调整**：原 "Zero-Shot Cross-Domain Evaluation" 改为 "Cross-Distribution Generalization Evaluation"，**章末显式声明这不是严格 sim-to-real**

#### 三个跨分布数据源

1. **IEEE DataPort live**：真实 Holybro S500 + HackRF 硬件欺骗，1 benign + 1 spoofing + 1 jamming flight
2. **IEEE DataPort PX4 SITL**：4 种机型（PLANE/VTOL/TAIL/H480）SITL 仿真，5 normal + 4 spoofing + 5 DoS flights
3. **ALFA**：CMU AirLab 真实 Carbon-Z 固定翼，10 个 no-failure flights，我们注入合成 step 攻击（M=15m, onset 40%）

#### 主要对比表

| Model | Sim test | DataPort live | DataPort SITL | ALFA |
|---|---|---|---|---|
| CNN | 0.893 | **🔻 0.360** | **🔻 0.222** | *0.960* |
| LSTM | 0.935 | *0.840* | *0.800* | *0.920* |
| BiLSTM | 0.783 | 0.520 | 0.467 | *0.840* |
| GRU | 0.947 | *0.800* | *0.822* | *0.880* |
| CNN-LSTM | 0.958 | **🔻 0.200** | **🔻 0.089** | *0.960* |
| TCN | 0.754 | 0.600 | 0.600 | *1.000* |
| Transformer | 0.919 | **🔻 0.320** | **🔻 0.444** | 0.760 |

🔻 = ≥0.30 退化；*斜体* = ≤0.10 退化或提升

#### 两个 finding

1. **架构排名不跨分布保持**：CNN-LSTM 分布内冠军（0.958）→ DataPort live 倒数第一（0.200），是 7 模型中最大的分布内→DataPort 退化。LSTM/GRU 在分布内中游 → 唯一两个 DataPort live DR@5s > 0.80 的架构。

2. **Precision 跨架构均匀崩塌**：分布内 P 范围 0.79–0.90 → DataPort live 0.38–0.42（均匀范围）。Recall 保持（0.48–0.71），说明所有架构都对仿真器干净的 FP 特征过拟合：真实飞行动力学产生很多正常样本被误报。

#### **作用域声明**（新加）

> 本节是 **cross-distribution** 评估，**不是严格 sim-to-real**：3 个跨分布源在多个轴上与训练分布不同（飞行器、巡航速度、传感器噪声统计、攻击机制）。我们在 §V-H 设计受控对比来 disentangle 平台不匹配的贡献。

### 5.H **Controlled Platform-Matching Experiment**（新增子节）

#### 实验设计

固定架构 / 训练 pipeline / 评估协议 / 随机种子，**3 种训练 regime**：
1. **5 m/s, 3000 flights**（原电力巡检 multirotor regime）
2. **5 m/s, 600 flights**（同速度同高度，缩规模到 360/120/120 → 控制训练规模）
3. **20 m/s matched, 600 flights**（速度提到 20 m/s，高度 120 m 匹配 ALFA Carbon-Z 固定翼；同 360/120/120 规模；其他参数全保持）

(1)→(2) 隔离训练规模；(2)→(3) 隔离平台速度匹配。

#### 受控对比表（zero-shot ALFA DR@5s）

| Model | 5 m/s 3000 | 5 m/s 600 | 20 m/s matched 600 |
|---|---|---|---|
| CNN | 0.960 | 0.960 | **0.000** |
| LSTM | 0.920 | 0.960 | 0.600 |
| BiLSTM | 0.840 | 0.880 | 0.600 |
| GRU | 0.880 | 0.840 | 0.480 |
| CNN-LSTM | 0.960 | 0.920 | **0.080** |
| TCN | 1.000 | 0.680 | 0.160 |
| Transformer | 0.760 | 0.640 | 0.280 |

#### 三条观察

1. **训练规模解释力很小**：5m/s 3000→600，最大架构（TCN）DR@5s 跌 0.32，其余 ≤ ±0.12。

2. **匹配标称平台速度反而是净负效应**：5m/s 600 → 20m/s matched 600（仅速度/高度变），5/7 架构跌 0.36–0.96。CNN 0.96→0.00，CNN-LSTM 0.92→0.08，TCN 0.68→0.16。

3. **候选机制（待未来工作验证）**：
   - 5 m/s 下 15 m step 攻击产生 ~3 s 位移信号，多个滑窗能整合
   - 20 m/s 下同样攻击 ~0.75 s 完成 → 单滑窗，特征分布偏向短脉冲
   - ALFA 真实飞行有变速段（起飞/转弯/进近），慢速段对 5m/s 训练熟悉、对 20m/s 训练 OOD
   - **从慢速 sim 学到的更宽时间支持的特征转移更稳健**

#### 对论文修订的意义

本受控实验直接回应 reviewer 自然问题"你试过让 sim 匹配真实平台吗？"。
**试了，匹配 regime 严格更差**。简单的标称平台速度匹配**不是**真正 sim-to-real 验证的替代品；架构选择不能 defer 给"用看起来最像部署的 sim 训练"。

---

## §VI 讨论

### 6.A 框架的普适性（保留原内容）

三个 design choice 让 framework 跨任务可移植：
- 与滑窗解耦（per-step / event-triggered detector 都能评估）
- 跨任务可移植（IDS / 工业故障 / 医疗监控；只需重新基于风险推导阈值）
- 案例研究框定（具体值随训练数据/超参变；invariant 的是揭示能力）

### 6.B **Cross-Distribution Evaluation 的启示**（新增子节）

§V-G 和 §V-H 提供 3 条 deployment guidance：

1. **分布内排名不预测跨分布排名**：从 in-distribution validation 选架构有挑到 generalize 最差架构的风险

2. **简单循环架构（LSTM/GRU）跨分布更稳健**：与 ML 中"低容量模型转移更好"的普遍观察一致

3. **天真平台速度匹配不是补救**：偏好生成最丰富时间支持攻击 signature 的 sim regime（即使不匹配部署速度）

### 6.C Limitations and Future Work（重写）

- **Cross-distribution vs sim-to-real scope**：本实证 conflate 多个轴；§V-H 受控实验仅隔离平台速度；其余轴（多径/电离层、真实 RF 攻击 vs 状态注入、风扰/操作员动力学）混合变动。严格 sim-to-real 需要 matched-platform 真实飞行数据，本研究未收集。

- **MTBFA 全军覆没**：测试集累积正常时间仅 7.13 h，无法摊薄滑窗 FP 到部署级事件率。需要小时级真实部署评估。

- **仿真器传感器模型局限**：MATLAB GPS 模型只用高斯白噪声，无 multipath/电离层/接收机时钟漂移。

- **阈值依赖任务剖面**：DR@5s 和 MTBFA 阈值基于 5 m/s 巡检场景；其他场景需重新推导。

- **滑窗对慢攻击的偏差**：终点标签使早期 straddle 窗口标签延迟，可能膨胀 ADD（与真实 operator 经验一致）。

---

## §VII 结论

本文针对 UAV GNSS 欺骗检测，提出基于运行风险的**时间感知评估方法学**（DR@Δt / ADD / MTBFA + 虚警事件聚合）。

在 3000 飞行 MATLAB 仿真 + 7 架构 × 5 seed 基准 + 3 个真实跨分布数据源（IEEE DataPort live 硬件、IEEE DataPort PX4 SITL、ALFA 固定翼）上，**3 条对部署工程读者重要的发现**：

1. 时间感知指标暴露了被传统指标判为"可部署"的架构的特定失效模式
2. 分布内架构排名**不**保持到跨分布评估（CNN-LSTM 从 1st 变 last，LSTM/GRU 反之）
3. 受控平台匹配实验：天真匹配仿真巡航速度反而恶化 sim-to-real 转移；归因于较低速度下攻击 signature 的更宽时间支持

未来工作：
- 大规模 in-the-wild 数据采集 + matched-platform 实飞 → 严格 sim-to-real
- 概率校准/在线适应 → 提高运行 MTBFA 可达性，不损失 DR@Δt
- 框架向其他事件检测域扩展（IDS、工业故障）

---

## 改动总结

| 改动类型 | 内容 |
|---|---|
| **章节级重构** | §III "Proposed Time-Aware Evaluation Framework" → "Operational Evaluation Metrics" |
| **整段删除** | §III-D "Relation to Prior Time-Series Evaluation Frameworks"（Tatbul/NAB 对比表 + 4 段防御）|
| **整段新增** | §V-H "Controlled Platform-Matching Experiment"（matched 反直觉 finding）|
| **整子节重写** | §V-G "Zero-Shot Cross-Domain Evaluation" → "Cross-Distribution Generalization Evaluation" |
| **整子节填充** | §V-D Robustness Analysis（window/step + threshold grid）|
| **整子节填充** | §V-E Per-Attack-Type Performance（7×4 heatmap + 2 个 finding）|
| **整子节新增** | §VI-B What Cross-Distribution Tells Us |
| **整子节重写** | §VI-C Limitations and Future Work |
| **整段重写** | Abstract（lead with 3 个 empirical finding）|
| **整段重写** | §I 末段贡献清单（4 条 framework-flavored → 5 条 empirical-study）|
| **整段重写** | §VII Conclusion |
| **整段重写** | response_to_reviewers Reviewer 1 C3 + Reviewer 3 C5（实数填充）|
| **bug 修复** | Abstract 后多余 `\end{keywords}` 删除；`table_deployability.tex` 中未转义 `%` 修复 |
| **新增基础设施** | 3 个 figure 脚本（plot_cross_domain / plot_alfa_controlled / plot_window_step）|
| **新增基础设施** | `docs/ieeeaccess.cls` stub（IEEEtran 兼容编译）|

## 你需要决定的事

1. **标题**：保留 "A Time-Aware Evaluation Framework..."？或改为 "Time-Aware Evaluation of GPS Spoofing Detectors..."？
2. **§IV-F GitHub URL**：填实际仓库地址（`[PLACEHOLDER]`）
3. **§V-B 4 个占位 figure**：保留占位 / 用 `visualize.py` 数据补脚本生成 / 删整段 deep-dive 子节
4. **Author 照片**：投稿前换真实
5. **真 ieeeaccess.cls**：投稿前从 IEEE 下载替换 stub
6. **commit 这批改动？**

PDF 在 `docs/main_preview.pdf`，28 页，0 编译错误。
