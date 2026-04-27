# QKD 光网络仿真

光纤量子密钥分发（QKD）网络的物理层噪声建模与网络层资源分配仿真平台。

## 物理层建模

- **FWM**（四波混频）噪声：离散信道模型 + 连续 PSD 模型
- **SpRS**（自发拉曼散射）噪声：离散信道模型 + 连续 PSD 模型
- **GN-model**（高非线性干扰）：Poggiolini GN-model Eq.1 + Eq.120

公式文档：`docs/formulas_*.md`

## 环境搭建

### 方式一：conda（推荐）

```bash
conda env create -f environment.yml
conda activate qkd_env
```

### 方式二：pip

```bash
pip install -e ".[plot]"      # 核心 + 绘图依赖（dash, plotly, openpyxl）
pip install -e ".[all]"        # 全部依赖（含 ILP / 强化学习）
pip install -e ".[gpu]"        # 额外安装 GPU 加速（需 CUDA）
```

## 快速开始

### 1. 噪声计算

```python
from qkd_sim.config.schema import FiberConfig, WDMConfig
from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import build_wdm_grid, SpectrumType
from qkd_sim.physical.noise.dispatcher import compute_noise

# 光纤配置（默认 SMF）
fiber_cfg = FiberConfig(
    alpha_dB_per_km=0.2,
    gamma_per_W_km=1.3,
    D_ps_nm_km=17.0,
    D_slope_ps_nm2_km=0.056,
    L_km=80.0,
    A_eff=80e-12,
    rayleigh_coeff=4.8e-8,
)

# WDM 配置（ITU-T G.694.1 C-band, 100 GHz 间隔）
wdm_cfg = WDMConfig(
    start_freq=190.1e12,
    start_channel=1,
    end_channel=61,
    channel_spacing=100e9,
    B_s=32e9,
    P0=1e-3,
    quantum_channel_indices=[38, 39, 40, 41],  # zero-based
)

fiber = Fiber(fiber_cfg)
grid = build_wdm_grid(wdm_cfg, spectrum_type=SpectrumType.RAISED_COSINE)

# 计算 FWM + SpRS 噪声
noise = compute_noise("all", fiber, grid)
print("FWM 前向噪声:", noise["fwm_fwd"])
print("SpRS 前向噪声:", noise["sprs_fwd"])
```

### 2. 绘图脚本

所有 Dash 脚本支持以下参数：

| 参数 | 说明 |
|------|------|
| `--type` | 噪声类型：`fwm`, `sprs`, `both`, `only_signal`, `with_signal` |
| `--modulation` | 调制格式：`16qam`（默认，Raised Cosine + OSA 16QAM），`ook`（NRZ-OOK + OSA OOK），`dp-16qam` |
| `--skr-model` | SKR 模型：`approx_finite`（默认），`infinite`，`strict_finite` |
| `--export-excel` | 预计算后导出 Excel 文件并退出（不启动 Dash 服务器） |
| `--export-only` | `--export-excel` 的简写 |

```bash
# App 1: 噪声 vs 光纤长度（端口 8050）
python scripts/plot_noise_dash_len.py --type=fwm --modulation=16qam
python scripts/plot_noise_dash_len.py --type=fwm --modulation=ook

# App 2: 噪声 vs 量子信道频率（端口 8051）
python scripts/plot_noise_dash_ch.py --type=fwm --modulation=16qam
python scripts/plot_noise_dash_ch.py --type=both --modulation=ook

# 导出 Excel 后直接退出
python scripts/plot_noise_dash_ch.py --type=fwm --export-only
# 生成: data/precomputed/noise_vs_frequency.xlsx  (每个 sheet = 一个光纤长度)
#       data/precomputed/simulation_report.txt  (仿真参数记录)

python scripts/plot_noise_dash_len.py --type=fwm --export-only
# 生成: data/precomputed/noise_vs_length.xlsx     (每个 sheet = 一个量子信道)
#       data/precomputed/simulation_report.txt

# 信号功率谱
python scripts/plot_signal_tx.py
```

启动后访问：
- App 1：http://localhost:8050
- App 2：http://localhost:8051

滑条可实时调整量子信道或光纤长度，噪声功率图同步更新。

### 3. YAML 配置文件

默认配置位于 `src/qkd_sim/config/defaults/`：

| 文件 | 内容 |
|------|------|
| `fiber_para/fiber_smf.yaml` | 标准单模光纤参数 |
| `wdm_para/wdm_100ghz.yaml` | ITU-T G.694.1 C-band 100 GHz 间隔 |
| `plot_para/model_comparison.yaml` | 绘图颜色/标签模型配置 |

### 4. 参数配置说明

#### 经典信道功率（统一功率）

修改 `wdm_para/wdm_100ghz.yaml` 中的 `P0`：
```yaml
P0: 1.0e-3   # 0 dBm = 1 mW/信道
P0: 2.0e-3   # 3 dBm = 2 mW/信道
```

#### 经典信道功率（逐信道不同）

通过代码在创建 `WDMConfig` 时传入 `channel_powers_W` 字典（单位：瓦特）：
```python
wdm_cfg = WDMConfig(
    ...,
    P0=1e-3,  # 默认功率
    channel_powers_W={38: 1.5e-3, 39: 2.0e-3, 40: 1.0e-3},  # 逐信道覆盖
)
```

#### 经典信道位置

修改 `wdm_para/wdm_100ghz.yaml` 中的 `classical_channel_indices`（zero-based 索引）：
```yaml
classical_channel_indices: [38, 39, 40]   # 默认：C39/C40/C41（193.9–194.1 THz）
classical_channel_indices: [30, 31, 32]    # C31/C32/C33（193.1–193.3 THz）
```

#### 调制格式

通过 `--modulation ook|16qam` 参数选择，不修改 YAML：
```bash
python scripts/plot_noise_dash_ch.py --type=fwm --modulation=ook    # OOK 场景
python scripts/plot_noise_dash_ch.py --type=fwm --modulation=16qam  # 16QAM 场景（默认）
```

#### 经典信道策略配置

默认使用 YAML 中的 `classical_channel_indices`（如 `[38, 39, 40]`），也可通过策略自动计算：

```bash
python scripts/plot_noise_dash_ch.py \
    --strategy-name=interleave \
    --num-classical=4 \
    --reference-channel=34
```

| 参数 | 说明 |
|------|------|
| `--strategy-name` | `equal_interval`（低频侧连续）或 `interleave`（半频间隔） |
| `--num-classical` | 经典信道数量 |
| `--reference-channel` | zero-based 参考信道索引（C35 = 34） |

保留约束（sync/reference 信道及其保护带宽）在 `wdm_100ghz.yaml` 的 `classical_channel_strategy.reserved` 中配置。

#### 光纤参数

修改 `fiber_para/fiber_smf.yaml` 中的任意字段（衰减系数 $\alpha$、非线性系数 $\gamma$、色散参数 $D$ 等）。

### 5. BB84 SKR 参数

BB84 安全码率配置统一管理于 `skr_para/bb84_config.yaml`，支持两种预设：

```bash
# 加载方式: load_skr_config(path, profile="custom")  # 默认
#           load_skr_config(path, profile="reference") # 文献参考值
```

| 分区 | 用途 | 主要差异 |
|------|------|---------|
| `custom` | 实际系统仿真（默认） | η_spd=0.1, IL=8dB, 非对称诱饵态/基矢 |
| `reference` | 复现文献结果 | η_spd=1.0, IL=0dB, 对称 BB84（Wiesemann et al.） |

#### SKR 绘图脚本

```bash
# SKR vs 距离（App 3）
python scripts/plot_skr_vs_distance.py --profile=custom      # 默认，实际系统参数
python scripts/plot_skr_vs_distance.py --profile=reference   # 文献参考值

# SKR 优化 vs 距离（App 4）
python scripts/plot_skr_optimization_vs_distance.py --profile=custom
python scripts/plot_skr_optimization_vs_distance.py --profile=reference

# 指定自定义 YAML（覆盖 --profile 默认路径）
python scripts/plot_skr_vs_distance.py --skr-config=/path/to/my_config.yaml --profile=custom
```

#### SKR 模型选择

噪声-vs-频率 / 噪声-vs-长度 Dash 脚本（`plot_noise_dash_ch.py` 和 `plot_noise_dash_len.py`）在 `--type=with_signal` 时会显示 SKR 子图。默认使用 **approx_finite**（近似有限长）模型。可通过 `--skr-model` 参数切换：

```bash
# 无限长密钥模型（无有限长效应，SKR 最高）
python scripts/plot_noise_dash_ch.py --type=with_signal --modulation=dp-16qam --skr-model=infinite

# 近似有限长模型（默认，3-state decoy + Gaussian 修正）
python scripts/plot_noise_dash_ch.py --type=with_signal --modulation=dp-16qam --skr-model=approx_finite

# 严格有限长模型（1-decoy + Hoeffding/Azuma 不等式，SKR 最低）
python scripts/plot_noise_dash_len.py --type=with_signal --modulation=dp-16qam --skr-model=strict_finite
```

三个模型的数学公式见 `docs/formulas_skr.md`，实现见 `src/qkd_sim/physical/skr/skr_decoy_bb84.py`。

#### 块长模式（Block Length）

SKR 计算支持两种块长指定方式，通过 `block_length` 字段配置：

```yaml
# 方式 A: 固定 Alice 发送脉冲数（默认）
block_length:
  mode: "alice"
  N_alice: 1.0e+7     # Alice 发送 10M 脉冲
  N_bob: ~

# 方式 B: 固定 Bob 检测事件数
block_length:
  mode: "bob"
  N_alice: ~            # mode=bob 时不使用
  N_bob: 1.0e+6        # Bob 目标检测 1M 事件
```

mode=alice 和 mode=bob **互斥**，不能同时指定。`mode=alice` 时 `N_bob` 必须为 `~`（null），反之亦然。

**物理含义**：
- `mode=alice`：Alice 固定发送 N_alice 个脉冲，Bob 检测数为随机变量，积分时间 t = N_alice / R_0
- `mode=bob`：Bob 固定检测 N_bob 个事件，Alice 发送脉冲数为随机变量，积分时间 t = N_bob / (R_0 · P_X_alice · P_X_bob · P_det)

## 项目结构

```
src/qkd_sim/
├── config/schema.py       # FiberConfig, WDMConfig 数据类
├── physical/
│   ├── fiber.py           # Fiber 类（损耗、色散、拉曼增益）
│   ├── signal.py          # WDMChannel, WDMGrid, SpectrumType
│   └── noise/
│       ├── base.py        # NoiseSolver ABC
│       ├── fwm_solver.py  # FWM 噪声（含 GPU 加速）
│       ├── sprs_solver.py # SpRS 噪声（离散 + 连续）
│       ├── gn_solver.py   # GN-model NLI
│       └── dispatcher.py  # 统一入口
├── network/               # 拓扑、路由、流量、资源分配
└── utils/
    ├── units.py           # 单位转换
    └── gpu_utils.py       # CuPy/NumPy 自动切换
scripts/
├── plot_noise_dash_len.py # Dash App 1
├── plot_noise_dash_ch.py  # Dash App 2
├── plot_signal_tx.py      # 信号功率谱
└── dash_utils.py          # 共享常量和辅助函数
docs/
├── formulas_signal.md     # 信号建模公式
├── formulas_fwm.md        # FWM 噪声公式
├── formulas_sprs.md       # SpRS 噪声公式
└── formulas_nonlinear.md  # GN-model 公式
```
