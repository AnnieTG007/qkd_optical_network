# QKD 与经典光信号共纤传输统一仿真平台

一个用于研究 QKD（量子密钥分发）与经典光通信信号共纤传输问题的科研仿真平台。

## 项目目标

构建**物理层噪声建模**和**网络层资源分配**的统一平台，支持：
- 物理层噪声评估（FWM + 自发/受激拉曼散射）
- 网络层资源分配（波长分配 + 纤芯分配）
- 面向实验与论文的可验证性

## 核心特性

### 物理层
- **FWM 模型**: 完整相位失配公式，矩阵运算优化
- **拉曼模型**: 基于 Mandelbaum 2003 Eq.(5)，查增益系数表（非横截面）
- **解析积分**: 仅需光纤末端 (L 处) 结果，无需 ODE 求解器
- **单芯/多芯**: 当前实现单芯（SMF/HCF），预留芯间噪声接口

### 网络层
- **模块化设计**: 业务生成、资源分配、状态管理、指标计算解耦
- **算法框架**: 支持启发式、强化学习（DQN）、ILP（预留）
- **协同度优化**: 经典性能与量子性能的几何平均

### 实验验证
- **OSA 数据导入**: 支持光谱仪导出格式
- **CSV 输出**: Origin 友好的数据格式 + 自动化脚本
- **模型对比**: 单频模型 vs 带宽模型

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本用法（物理层）

```python
from physics.signal import WDMChannel
from physics.fiber import Fiber
from physics.noise import fwm, raman
from propagation.solver import SingleSpanSolver

# 定义 WDM 信道
channels = [
    WDMChannel(center_freq=193.4e12, power=1e-3, baudrate=32e9, roll_off=0.1, modulation='QPSK'),
    # ... 更多信道
]

# 定义光纤参数
fiber = Fiber(fiber_type='SMF', length=50e3)

# 计算噪声
solver = SingleSpanSolver(fiber, channels)
noise_psd = solver.compute_noise()  # 返回 PSD [W/Hz]

# 计算 QKD 信道噪声（在接收机带宽内积分）
qkd_noise = solver.integrate_noise_over_bandwidth(
    qkd_center_freq=193.4e12,
    bandwidth_nm=0.12
)
```

### 基本用法（网络层）

```python
from network.topology import Topology
from network.traffic import TrafficGenerator
from network.allocation import ResourceAllocator
from network.metrics import MetricsCalculator

# 定义网络拓扑
topo = Topology.from_adjacency_matrix([...])

# 生成业务流
traffic_gen = TrafficGenerator(arrival_rate=50, hold_time=4)
events = traffic_gen.generate(num_events=1000)

# 资源分配
allocator = ResourceAllocator(algorithm='heuristic')
allocations = allocator.allocate(events, topo)

# 计算指标
metrics = MetricsCalculator()
blocking_rate = metrics.blocking_rate(events, allocations)
skr = metrics.quantum_key_rate(...)
synergy = metrics.synergy_score(blocking_rate, skr)
```

---

## 项目结构

```
qkd_optical_network/
├── README.md                    # 本文件
├── FORMULAS_REVISION.md         # 公式列表与修订记录
├── requirements.txt             # Python 依赖
│
├── constants/
│   ├── __init__.py
│   └── fiber_parameters.py      # SMF/HCF 标准参数
│
├── physics/
│   ├── __init__.py
│   ├── signal.py                # WDMChannel, SignalState
│   ├── fiber.py                 # Fiber 类
│   └── noise/
│       ├── __init__.py
│       ├── fwm.py               # FWM 噪声模型
│       ├── raman.py             # 拉曼噪声模型
│       └── inter_core.py        # 芯间噪声（预留）
│
├── propagation/
│   ├── __init__.py
│   ├── solver.py                # 单跨段噪声求解器
│   └── effective_length.py      # 有效长度计算
│
├── network/
│   ├── __init__.py
│   ├── topology.py              # 网络拓扑
│   ├── traffic.py               # 业务生成
│   ├── allocation.py            # 资源分配
│   ├── state.py                 # 网络状态
│   └── metrics.py               # 指标计算
│
├── optimization/
│   ├── __init__.py
│   ├── base_optimizer.py        # 优化器基类
│   ├── heuristic.py             # 启发式算法
│   └── rl/                      # 强化学习（预留）
│
├── evaluation/
│   ├── __init__.py
│   ├── scenario.py              # 实验场景配置
│   ├── csv_export.py            # Origin 输出
│   └── comparison.py            # 单频 vs 带宽模型对比
│
└── tests/
    ├── __init__.py
    ├── test_fwm.py
    └── test_raman.py
```

---

## 公式文档

所有物理公式详见 [`FORMULAS_REVISION.md`](./FORMULAS_REVISION.md)。

如需修改或添加公式，请直接编辑该文件。

---

## 单位约定

| 物理量 | 单位 | 说明 |
|--------|------|------|
| 频率 | Hz | 所有频率统一使用 Hz，不使用 THz |
| 功率 | W | 所有功率统一使用 W，不使用 dBm |
| 长度 | m | 光纤长度、波长等 |
| 时间 | s | 门控时间、脉冲重复周期等 |
| 温度 | K | 绝对温度 |

**单位转换工具**: `constants物理常数.py` 提供常用转换函数

---

## 参考代码

| 功能 | 参考代码路径 | 说明 |
|------|-------------|------|
| 拉曼噪声 | `C:\Users\Annie\Desktop\reference\tool.py` | 主要参考，查增益系数方式 |
| FWM 噪声 | `E:\...\DQN-QKD\MCF.py` | 组合枚举逻辑参考 |
| 光纤参数 | `E:\...\HCF_Optical_Network\Optical_Fiber.py` | 参数定义参考 |
| 网络层 | `E:\...\DQN-QKD\environments\qkd_env\` | 解耦重构 |

---

## 版本历史

| 版本 | 日期 | 内容 |
|------|------|------|
| 0.1.0 | 2026-03-25 | 初始版本，Phase 0 基础框架 |

---

## 许可

本项目为科研代码，仅供内部使用。

---

## 联系方式

- 项目 PI: [待填写]
- 主要开发者: [待填写]
- 问题反馈: [待填写]
