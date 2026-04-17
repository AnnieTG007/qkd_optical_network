# QKD 光网络仿真

光纤量子密钥分发（QKD）网络的物理层噪声建模与网络层资源分配仿真平台。

## 物理层建模

- **FWM**（四波混频）噪声：离散信道模型 + 连续 PSD 模型
- **SpRS**（自发拉曼散射）噪声：离散信道模型 + 连续 PSD 模型
- **GN-model**（高非线性干扰）：Poggiolini GN-model Eq.1 + Eq.120

公式文档：`docs/formulas_*.md`

## 环境搭建

### 依赖环境

```bash
conda create -n qkd_env python>=3.10
conda activate qkd_env

# 核心依赖
pip install numpy scipy matplotlib networkx pyyaml dash plotly

# 可选：GPU 加速（需要 CUDA）
pip install cupy

# 可选：ILP 求解器 / 强化学习
pip install pulp
pip install gymnasium

# 安装本项目
pip install -e .
```

### conda 环境（已验证）

```
C:\Users\Annie\miniconda3\envs\qkd_env\python.exe
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

所有 Dash 脚本使用统一的 `--type` 参数：

```bash
# App 1: 噪声 vs 光纤长度（端口 8050）
python scripts/plot_noise_dash_len.py --type=fwm
python scripts/plot_noise_dash_len.py --type=sprs
python scripts/plot_noise_dash_len.py --type=both       # FWM + SpRS 功率叠加
python scripts/plot_noise_dash_len.py --type=with_signal  # 包含信号 TX

# App 2: 噪声 vs 量子信道频率（端口 8051）
python scripts/plot_noise_dash_ch.py --type=fwm
python scripts/plot_noise_dash_ch.py --type=sprs
python scripts/plot_noise_dash_ch.py --type=both
python scripts/plot_noise_dash_ch.py --type=with_signal

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

## 核心参数

### WDM 信道频率（ITU-T G.694.1）

频率公式：`f(n) = 190.1 THz + (n - 1) × 100 GHz`

| 信道 | 频率 |
|------|------|
| C01 | 190.1 THz |
| C39 | 193.9 THz |
| C61 | 196.1 THz |

### SpRS B_noise 参数

离散 SpRS 模型的噪声收集带宽 `B_noise`：
- **默认 20 GHz**
- 与 `channel_spacing`（100 GHz）和信号带宽 `B_s`（32 GHz）独立
- 可通过 `DiscreteSPRSSolver(noise_bandwidth_hz=xxx)` 配置

## 公式修正记录

| 项目 | 修正 |
|------|------|
| `alpha_dB_km_to_per_m` | ×1e-3（原误写 1e3） |
| FWM 连续前向系数 | γ²/9（非 4γ²/9） |
| SpRS 噪声带宽 | `bandwidth`→`B_noise`（非泵浦-信号频移 Δf） |
| GN-model | Poggiolini Eq.1 + Eq.120 |

详见 `docs/formulas_*.md` 和 `.claude/devjournal.md`。

## 注意事项

- `.claude/` 目录为本地文档（不在 GitHub 同步）
- `.cupy_cache/` 为 CuPy 编译缓存，已加入 `.gitignore`
- 项目指南同时维护于 `CLAUDE.md`（面向 Claude）和 `AGENTS.md`（面向 Codex），两者同步更新
