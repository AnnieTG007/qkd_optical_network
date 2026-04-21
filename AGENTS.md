# QKD 光网络仿真 — Codex 项目指南

## A. 项目目标

- **物理层**：光纤非线性噪声建模（FWM/SpRS/GN-model）、功率谱绘制
- **网络层**：业务生成、路径规划、频谱资源分配（ILP/启发式/RL）
- **核心原则**：所有结果可由脚本复现；公式文档与代码一一对应；参数配置与代码解耦
- **Git 管理**：每次修改调试通过后同步到 GitHub

## B. 架构概览

详细结构定义见 [architecture.md](architecture.md)。

```
src/qkd_sim/
├── config/          # FiberConfig, WDMConfig (YAML + dataclass)
├── physical/
│   ├── fiber.py     # Fiber 类
│   ├── signal.py    # WDMChannel, WDMGrid, SpectrumType
│   ├── noise/       # NoiseSolver ABC, SpRS/FWM/GN solvers, dispatcher
│   └── spectrum.py  # 绘图函数
├── network/         # topology, routing, traffic, resource, algorithms
└── utils/units.py   # 单位转换
scripts/             # Dash apps + 绘图脚本
dash_utils.py        # 所有 Dash 脚本共享常量和辅助函数
data/
├── precomputed/     # 预计算缓存 CSV + Excel 导出
└── osa/             # OSA 实测频谱数据
environment.yml      # conda 环境定义（引用 pyproject.toml）
```

## C. 核心约束

| 约束 | 说明 |
|------|------|
| 变量命名 | 严格对应 `docs/parameters.md` "代码变量名"列 |
| 物理模型可追溯 | 所有模型对应 docs/ 下的公式文档 |
| 单位统一 | 频率 Hz、波长 m、功率 W、损耗 1/m 或 dB/km |
| 优先向量化 | 噪声计算向量化，减少 for 循环 |
| 参数与代码解耦 | YAML 配置文件 + dataclass(__post_init__) |
| 物理常数 | `from scipy.constants import h, k, c` |

## D. 公式文档

| 文档 | 内容 |
|------|------|
| `docs/formulas_signal.md` | 信号建模（G_TX 升余弦/OS A） |
| `docs/formulas_fwm.md` | FWM 噪声公式 |
| `docs/formulas_sprs.md` | SpRS 噪声公式 |
| `docs/formulas_nonlinear.md` | GN-model NLI 公式（Poggiolini Eq.1/120） |
| `docs/parameters.md` | 参数定义与单位 |

## E. 测试规则

- 每个模型：维度检查 + 极限情况 + 与文献结果比对
- 每次修改优先跑对应模块测试，不全项目盲跑
- 公式未确认时保留为 `experimental`

## F. 对 Codex 的行为约束

- 任意模块改动，先给计划，等用户确认后再实施
- 改动时说明影响范围；不大规模重构无关代码
- 新增代码带 docstring、类型标注、测试

---

## G. WDM 信道频率规范（ITU-T G.694.1）

**公式**：`f(n) = start_freq + (n - start_channel) × spacing`

| 参数 | 值 |
|------|-----|
| `start_freq` | 190.1 THz（C01） |
| `start_channel` | 1 |
| `end_channel` | 61（C61） |
| `channel_spacing` | 100 GHz |

**CLASSICAL_INDICES 从 YAML 读取（不再是硬编码）**：
- 位置：`wdm_para/wdm_100ghz.yaml` 的 `classical_channel_indices` 字段
- 统一使用 **1-based ITU G.694.1 信道编号**（C01=190.1THz，C61=196.1THz）
- `classical_channel_indices` 有值时优先使用；为空/None 时由 `classical_channel_strategy` 策略生成
- 正确：`[39, 40, 41]` → C39(193.9), C40(194.0), C41(194.1) THz

**经典信道策略配置（classical_channel_indices 为空时生效）**：
- `dash_utils.py` 中的 `resolve_classical_indices()` 实现

| 字段 | 说明 |
|------|------|
| `name` | `"equal_interval"`（低频侧连续）或 `"interleave"`（半频间隔） |
| `reference_channel` | 1-based ITU G.694.1 参考信道号（e.g., 35 = C35 = 193.5 THz，自身为量子信道） |
| `num_classical` | 经典信道数量 |
| `reserved[].name` | 保留区域标识符（`"sync"`、`"reference"`） |
| `reserved[].channel` | 保留信道 1-based ITU 信道号 |
| `reserved[].bandwidth_ghz` | 保留带宽 [GHz]，保护区间为 `(f - bw/2, f + bw/2)` |

**equal_interval**：候选 = `[ref-N, ref-N+1, ..., ref-1]`，取低频侧 N 个
**interleave**：候选 = `[ref + (2*i-(N-1))/2.0 for i in range(N)]`，支持半整数 ITU 索引
- N=4 → 偏移 `[-1.5, -0.5, 0.5, 1.5]` → ITU `[32.5, 33.5, 34.5, 35.5]`
- N=3 → 偏移 `[-1.0, 0.0, 1.0]` → ITU `[33.0, 34.0, 35.0]`（对称整数）

**保护约束**：候选经典信道 passband `(f_c-50, f_c+50) GHz`
与保留区间 `(f_r-bw/2, f_r+bw/2) GHz` 不得有交集；保留信道自身永不为经典信道
- 若有效位置不足 N 个或超出 `[1, 61]` 范围则 `ValueError`

---

## H. 绘图代码规范

**所有 Dash 脚本必须从 `scripts/dash_utils.py` 导入公共代码**，禁止重复定义。

### 两个统一 Dash App

| 文件 | 端口 | 滑条 | X 轴 |
|------|------|------|------|
| `plot_noise_dash_len.py` | 8050 | 量子信道 | 光纤长度（km） |
| `plot_noise_dash_ch.py` | 8051 | 光纤长度 | 量子信道频率（THz） |

**参数**：
- `--type`：`fwm` / `sprs` / `both` / `only_signal` / `with_signal`
- `--modulation`：`16qam`（默认，Raised Cosine + OSA 16QAM）/ `ook`（NRZ-OOK + OSA OOK）
- `--export-excel` / `--export-only`：导出 Excel 后退出（不启动服务器）

详细规范见 [.Codex/plotting.md](plotting.md)。

---

## I. 已确认公式修正

| 项目 | 修正 |
|------|------|
| `alpha_dB_km_to_per_m` | 乘以 1e-3（原误写 1e3） |
| FWM 连续前向系数 | γ²/9（非 4γ²/9） |
| FWM 后向 F(l) | z 为观测位置 |
| GN-model | Poggiolini Eq.1 + Eq.120（2026-04） |

详细修正记录见 [.Codex/devjournal.md](devjournal.md)。

---

## J. GPU 加速

`src/qkd_sim/utils/gpu_utils.py` 提供 CuPy/NumPy 自动切换：
- `GPU_ENABLED`：全局标志
- `get_array_module()`：返回 `cupy` 或 `numpy`
- `to_device()` / `to_host()`：CPU↔GPU 数据迁移
- 无 CUDA 时静默回退到 NumPy，不影响 CPU 环境

FWM 求解器 (`fwm_solver.py`) 在 GPU 可用时自动使用 `_gpu_fwm_efficiency_vec`。

---

## K. SpRS B_noise 参数

离散 SpRS 模型的噪声收集带宽 `B_noise` 与 `channel_spacing` 独立：
- **默认 20 GHz**（可配置）
- 用于拉曼截面公式中替代旧错误的 `Δf`（泵浦-信号频移）

---

## L. 虚拟环境

**conda 环境**：`C:\Users\Annie\miniconda3\envs\qkd_env\python.exe`

重建命令见 [.Codex/devjournal.md](devjournal.md)。

---

## M. 文档同步

`AGENTS.md`（Codex 项目指南）与本文件 `AGENTS.md` 为镜像文档，内容应同步更新：
- 修改 AGENTS.md 后，同步更新 AGENTS.md 的对应章节
- 两者结构一致，仅标题和面向对象不同（Codex → Codex）
