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

**CLASSICAL_INDICES 必须使用 zero-based 索引**：
- 正确：`[38, 39, 40]` → C39(193.9), C40(194.0), C41(194.1) THz
- 错误：`[39, 40, 41]` → 偏移到 C40, C41, C42

---

## H. 绘图代码规范

**所有 Dash 脚本必须从 `scripts/dash_utils.py` 导入公共代码**，禁止重复定义。

### 两个统一 Dash App

| 文件 | 端口 | 滑条 | X 轴 |
|------|------|------|------|
| `plot_noise_dash_len.py` | 8050 | 量子信道 | 光纤长度（km） |
| `plot_noise_dash_ch.py` | 8051 | 光纤长度 | 量子信道频率（THz） |

`--type` 参数：`fwm` / `sprs` / `both` / `only_signal` / `with_signal`

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
