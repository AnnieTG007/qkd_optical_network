# CLAUDE.md - QKD 与经典光信号共纤传输统一仿真平台

**重要**: 本文件用于保存项目上下文，确保在不同会话间可以无缝继续工作。

---

## 项目目标

构建**QKD 与经典光信号共纤传输统一仿真平台**：
- 物理层噪声建模：FWM（四波混频）、自发/受激拉曼散射
- 网络层资源分配：多目标优化（经典 BER + QKD 密钥率协同）
- 面向实验与论文的可验证性：OSA 数据导入、CSV 输出、模型对比

---

## 当前进度

### Phase 0 ✅ 已完成
- [x] 项目目录结构
- [x] README.md + FORMULAS_REVISION.md（公式文档）
- [x] requirements.txt
- [x] constants 模块（SSMF/HCF 参数、GNpy 拉曼系数表）
- [x] physics/signal.py（WDMChannel、SignalState、OSA 导入）
- [x] physics/fiber.py（Fiber 类、MultiCoreFiber 预留）
- [x] physics/noise/fwm.py（FWM 噪声计算）
- [x] physics/noise/raman.py（拉曼噪声计算，Mandelbaum 2003 Eq.5）
- [x] physics/noise/inter_core.py（芯间噪声预留）

### Phase 1 ✅ 已完成
- [x] propagation/solver.py（单跨段噪声求解器 SingleSpanSolver）
- [x] propagation/effective_length.py（有效长度计算器）
- [x] tests/test_fwm.py（FWM 单元测试，7 项测试全部通过）
- [x] tests/test_raman.py（拉曼单元测试，8 项测试全部通过）
- [x] FWM 效率公式更新为 FORMULAS_REVISION.md 公式 (2)（考虑波长依赖性衰减）

### Phase 1+ ✅ 已完成（补充）
- [x] validation/compute_noise_spectrum.py（噪声功率谱计算脚本，参考 tool.py 风格）
- [x] tests/README_TESTS.md（测试详细说明文档）

### Phase 2 ✅ 已完成（连续带宽模型）
- [x] physics/noise/fwm_continuous.py（连续带宽 FWM 模型）
- [x] physics/noise/raman_continuous.py（连续带宽 Raman 模型）
- [x] SpectrumType enum（SINGLE_FREQ, RECTANGULAR, RAISED_COSINE, OSA_SAMPLED）
- [x] WDMChannel.get_psd()（多种频谱形状支持，解析归一化）
- [x] WDMChannel.get_capacity()（QAM 调制容量计算，公式 (29)-(31)）
- [x] validation/model_comparison/compare_noise_models.py（离散 vs 连续噪声模型对比）
- [x] validation/model_comparison/compare_spectrum_models.py（频谱形状对比）
- [x] Raman 连续模型验证通过（与离散模型 0 dB 差异）
- [x] FWM 连续模型验证通过（考虑带宽内额外 FWM 产物）

### Phase 2.5 ✅ 已完成（信号模型统一）
- [x] signal.py PSD 解析归一化统一（公式 (27)-(28)）
- [x] 删除 get_psd_analytical() 冗余方法
- [x] FORMULAS_REVISION.md 添加信号模型章节（公式 (25)-(31)）
- [x] CLAUDE.md 添加修改规范（公式 - 代码一致性、验证脚本重跑）

### Phase 2.6 ✅ 已完成（连续频谱可视化）
- [x] validation/spectrum_computation/compute_continuous_spectrum.py（连续噪声功率谱 + 链路信号功率谱计算与输出）
- [x] 每采样点精确计算 PSD（`WDMChannel.get_psd(freq_array)` 逐点求值）
- [x] 噪声功率谱基于 `return_spectrum=True` 向量化连续模型
- [x] 输出格式：每场景 10 张 PNG（2 组 × 5 张）+ 1 个 CSV

---

## 关键技术决策

### 光纤类型
- **仅使用 `SSMF` 和 `HCF`**，已移除 `SMF`（统一用 `SSMF` 表示标准单模光纤）

### 噪声模型
- **FWM**: 完整相位失配公式（考虑色散斜率），矩阵运算优化
- **拉曼**: 基于 Mandelbaum 2003 Eq.(5)，查 GNpy 增益系数表（非横截面）
- **泵浦不耗尽近似**: SRS 简化模型（后续可迭代泵浦耗尽效应）
- **解析积分**: 只需光纤末端 (L 处) 结果，无需 ODE 求解器

### 单位约定
| 物理量 | 单位 |
|--------|------|
| 频率 | Hz |
| 功率 | W |
| 长度 | m |
| 衰减 | m⁻¹ |
| 非线性系数 | W⁻¹·m⁻¹ |
| 色散系数 | s/m² |

### 协同度公式
$$协同度 = \sqrt{I_{classical} \cdot I_{quantum}}$$

---

## 待确认事项

无。所有待办事项已完成。

---

## 当前工作上下文

### 当前关注模块
- **validation/spectrum_computation/** - 频谱计算类脚本
  - `compute_noise_spectrum.py`：离散模型噪声功率谱（按信道中心频率）
  - `compute_continuous_spectrum.py`：连续带宽模型功率谱（每采样点精确计算）

### 当前关键问题
1. **FWM 连续模型 vs 离散模型差异**（已记录）
   - Channel 1（中心）: ~0 dB 差异 ✓
   - Channel 2（边带）: +13.77 dB 差异（连续 > 离散）
   - 原因：连续模型捕获了带宽内额外的 FWM 组合

2. **Raman 连续模型验证通过** - 离散 vs 连续：0.00 dB 差异 ✓

### 代码执行提示
```bash
# 连续频谱计算（新）
python validation/spectrum_computation/compute_continuous_spectrum.py

# 离散噪声功率谱（原有）
python validation/spectrum_computation/compute_noise_spectrum.py

# 噪声模型对比
python validation/model_comparison/compare_noise_models.py

# 单元测试
python -m pytest tests/ -v
```

---

## 文件结构

当前完整文件结构:

```
qkd_optical_network/
├── .gitignore                # Git 忽略文件
├── CLAUDE.md                 # 本文件 (上下文管理)
├── README.md                 # 项目说明
├── FORMULAS_REVISION.md      # 公式文档 (公式 (1)-(24))
├── requirements.txt
├── constants/
│   ├── __init__.py
│   └── fiber_parameters.py   # SSMF/HCF 参数、GNpy 拉曼表
├── physics/
│   ├── __init__.py
│   ├── signal.py             # WDMChannel, SignalState, OSA 导入
│   ├── fiber.py              # Fiber 类，MultiCoreFiber 预留
│   └── noise/
│       ├── __init__.py
│       ├── fwm.py            # FWM 噪声 (离散模型)
│       ├── fwm_continuous.py # FWM 噪声 (连续带宽模型)
│       ├── raman.py          # 拉曼噪声 (离散模型)
│       ├── raman_continuous.py # 拉曼噪声 (连续带宽模型)
│       └── inter_core.py     # 芯间噪声 (预留)
├── propagation/              # Phase 1 已实现
│   ├── __init__.py
│   ├── solver.py             # SingleSpanSolver 单跨段求解器
│   └── effective_length.py   # 有效长度计算器 (公式 (11))
├── tests/                    # Phase 1 已实现
│   ├── __init__.py
│   ├── test_fwm.py           # FWM 测试 (7 项通过)
│   ├── test_raman.py         # 拉曼测试 (8 项通过)
│   └── README_TESTS.md       # 测试详细说明文档
├── validation/               # 验证脚本
│   ├── __init__.py
│   ├── model_comparison/     # 模型对比类脚本
│   │   ├── compare_noise_models.py    # 离散 vs 连续噪声模型
│   │   └── compare_spectrum_models.py # 频谱形状对比
│   ├── signal_validation/    # 信号验证类脚本
│   │   └── test_signal_model.py       # 信号模型测试
│   └── spectrum_computation/ # 频谱计算类脚本
│       ├── compute_noise_spectrum.py       # 离散模型噪声功率谱
│       ├── compute_continuous_spectrum.py  # 连续模型信号/噪声功率谱（新）
│       └── compare_raman.py               # 拉曼对比
├── output/                   # 计算结果输出目录
│   ├── noise_models/         # 噪声模型对比结果
│   ├── spectrum_models/      # 频谱模型对比结果
│   ├── noise_spectra/        # 噪声功率谱 CSV（离散模型）
│   └── scenarios/            # 场景仿真结果
│       ├── 1_classic_channel/    # 离散模型输出
│       ├── 3_classic_channels/   # 离散模型输出
│       └── 5_classic_channels/   # 离散模型输出
│   ├── 1_classic_channel/    # 连续模型输出（compute_continuous_spectrum.py）
│   │   ├── link_signal_spectrum_W.png        # 信号功率谱 [W 对数]
│   │   ├── link_signal_spectrum_dBm.png      # 信号功率谱 [dBm]
│   │   ├── noise_fwm_W.png                   # FWM 噪声 [W 对数]
│   │   ├── noise_fwm_dBm.png                 # FWM 噪声 [dBm]
│   │   ├── noise_sprs_W.png                  # SpRS 噪声 [W 对数]
│   │   ├── noise_sprs_dBm.png                # SpRS 噪声 [dBm]
│   │   ├── noise_fwm_sprs_W.png              # FWM+SpRS 总噪声 [W 对数]
│   │   ├── noise_fwm_sprs_dBm.png            # FWM+SpRS 总噪声 [dBm]
│   │   ├── noise_combined_W.png              # 信号+总噪声叠加 [W 对数]
│   │   ├── noise_combined_dBm.png            # 信号+总噪声叠加 [dBm]
│   │   ├── continuous_spectrum_data.csv      # 完整数据表
│   │   └── discrete_model/                  # 旧离散模型输出（归档）
│   ├── 3_classic_channels/   # 同上
│   ├── 5_classic_channels/   # 同上
│   ├── 8_classic_channels/
│   │   └── discrete_model/                  # 旧离散模型输出（归档）
│   └── statistics/
│       └── noise_model_comparison.csv        # 噪声模型对比统计
├── network/                  # 待实现 (Phase 3)
└── evaluation/               # 待实现 (Phase 2+)
```

---

## 下一步行动

### Phase 2+ : evaluation 模块 (QKD-经典协同评估)

#### 优先级 1：基础评估工具
- [ ] `evaluation/basics.py` - OSNR/SNR/Q 值计算
- [ ] `evaluation/classical_ber.py` - 经典系统 BER 计算

#### 优先级 2：QKD 密钥率
- [ ] `evaluation/qkd_key_rate.py` - BB84 协议密钥率
- [ ] `evaluation/cooperation_degree.py` - 协同度计算

#### 优先级 3：结果导出
- [ ] `evaluation/csv_export.py` - Origin 格式导出
- [ ] `evaluation/paper_plots.py` - 论文标准图表

---

### Phase 3 : network 模块 (网络层资源分配)
- [ ] `network/topology.py` - 网络拓扑定义
- [ ] `network/resource_allocation.py` - 资源分配算法
- [ ] `network/multi_objective_optimization.py` - 多目标优化

---

## 远期计划

### Phase 2+ : evaluation 模块 (QKD-经典协同评估)

#### 2.1 基础评估模块
- [ ] `evaluation/basics.py` - 基本物理量计算（OSNR, SNR, Q 值）
- [ ] `evaluation/classical_ber.py` - 经典通信 BER 计算
  - 公式：BER = 0.5 * erfc(Q / sqrt(2))
  - Q = (I₁ - I₀) / (σ₁ + σ₀)

#### 2.2 QKD 密钥率模块
- [ ] `evaluation/qkd_key_rate.py` - QKD 密钥率计算
  - PNS 攻击模型（decoy state protocol）
  - GLLP 公式：SKR ≤ qμνt₁(1 - H₂(e₁) - f₂H₂(e₂))
  - BB84 协议实现

#### 2.3 协同度评估
- [ ] `evaluation/cooperation_degree.py` - 协同度计算
  - 公式：协同度 = sqrt(I_classical × I_quantum)
  - Pareto 前沿分析

#### 2.4 结果导出
- [ ] `evaluation/csv_export.py` - Origin 格式导出
- [ ] `evaluation/visualization.py` - 结果可视化

### Phase 3 : network 模块 (网络层资源分配)

#### 3.1 网络拓扑
- [ ] `network/topology.py` - 拓扑定义
  - 节点模型（EDFA, OEO, OXC）
  - 链路模型（光纤段、放大器）
  - 拓扑发现与验证

#### 3.2 资源分配
- [ ] `network/resource_allocation.py` - 波长/功率分配
  - RWA（波长路由分配）
  - 功率优化算法
  - QKD/经典容量权衡

#### 3.3 多目标优化
- [ ] `network/multi_objective_optimization.py` - 多目标优化
  - 目标函数：
    - 最小化经典系统 BER
    - 最大化 QKD 密钥率
    - 最小化总功耗
  - 算法：NSGA-II, MOEA/D

### Phase 4 : 实验验证与论文支持

#### 4.1 OSA 数据处理
- [ ] `experiment/osa_import.py` - OSA 测量数据导入
  - 支持 .csv, .txt 格式
  - 波长/频率转换
  - 功率归一化

#### 4.2 模型校准
- [ ] `experiment/calibration.py` - 仿真参数校准
  - 色散系数校准
  - 损耗系数校准
  - 非线性系数校准

#### 4.3 论文图表生成
- [ ] `experiment/paper_plots.py` - 论文标准图表
  - 噪声功率谱图
  - 信道容量图
  - 协同度热力图
  - BER vs 距离曲线

### 远期待定功能
- 芯间拉曼效应（inter_core.py 预留接口）
- 泵浦耗尽模型（当前使用不耗尽近似）
- 多跨段累积效应
- 偏振模色散（PMD）建模

---

## 用户偏好
- 代码风格：简洁、科研可追溯性
- 公式假设：明确标注来源、单位、适用范围
- 接口设计：类接口优先（而非纯函数式）
- 扩展性：预留芯间噪声接口，但不提前实现

---

## 修改规范（重要）

每次修改项目时，必须遵守以下规则：

### 1. 公式 - 代码一致性
- **如果修改了 FORMULAS_REVISION.md 中的公式**，必须相应调整代码实现
- **如果修改了噪声/信号计算代码**，必须在公式文档中添加或更新对应公式
- 公式编号需在代码注释中引用（如 `# 参考公式 (25)`）

### 2. 验证脚本重跑
- **任何噪声计算模块的修改**后，`validation/` 下的所有验证脚本必须重新执行
- `output/` 目录中的所有结果图表需重新生成，确保与最新代码一致

### 3. 清理冗余
- 每次修改后检查是否有：
  - 未使用的 import
  - 重复的代码逻辑
  - 过时的 validation 输出文件
  - 废弃的方法（应标记 deprecation 而非直接删除）

### 4. SignalState 与 WDMChannel 功率分离
- `SignalState.powers` 存储沿光纤位置的功率演化
- `WDMChannel.power` 存储信道的标称功率
- 避免直接修改 `WDMChannel.power` 来计算不同位置的 PSD（需使用参数覆盖方式）

---

## 重要参考代码路径
- 拉曼模型：`tool.py`（查增益系数表 + Mandelbaum 公式 5）
- FWM 模型：`MCF.py` 中的 `get_four_wave_mixing`
- 密钥率：`SKR_new.py`

---

**最后更新**: 2026-03-29 (Phase 2.6 完成：连续频谱可视化脚本)
