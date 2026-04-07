QKD光网络科研项目仿真

## A. 项目目标
- 本项目用于博士生科研项目，用于实现物理层光纤非线性噪声建模计算、噪声功率谱绘制和网络层路径规划和频谱资源分配。
- 物理层需要计算量子信道、经典信道所受的噪声功率大小，并绘制信号功率谱图、量子信道FWM噪声功率谱图、量子信道SpRS噪声功率谱图、量子信道FWM+SpRS噪声功率谱图、经典信道非线性噪声功率谱图、经典信道功率+经典信道非线性噪声+量子信道FWM+SpRS噪声功率谱图，每个图都包含以W为单位和dBm为单位的两个版本图；
- 网络层需要实现业务生成和算法解耦，网络层可以兼容ILP、启发式和强化学习资源分配算法，令算法模块返回分配的波长资源即可；
- 将项目维护到github中，每次更改调试通过后需要同步到github上，仅包括公式文档和核心代码；
- 本项目的所有核心特征都必须体现在CLAUDE.md中，如果CLUADE.md超过两百行，则把具体细节以markdown文档形式写入.claude/目录下，并引用至CLAUDE.md。

## B. 架构约束

- 项目全部基于python语言实现，代码需保持较好的可读性，注释规范；
- 物理层噪声计算部分偏好向量化，减少for循环语句的使用以提高计算效率；
- 所有物理模型必须可追溯到文献、教材或我提供的笔记，保持计算式与公式一一对应；
- 各模块之间通过类或函数的规范接口通信，保持各模块的功能独立，包括：物理层噪声，网络层资源分配代码与物理层噪声模型解耦，
参数设置与代码解耦；
- 所有核心结果都可以由脚本复现实验。

## C. 数学与实现规则

### a) 物理层噪声模型公式文档
- 信号建模公式：docs/formulas_signal.md
- 量子信道受FWM噪声公式：docs/formulas_fwm.md  
- 量子信道受SpRS噪声公式：docs/formulas_sprs.md
- 经典信号噪声公式由GN-model计算，公式记于docs/formulas_nonliear.md
- 参数定义与单位：docs/parameters.md

实现某个模块前，先阅读对应的公式文档。

### b) 物理层实现规则
- 单位统一：频率 Hz、波长 m、功率 W、损耗 1/m 或 dB/km 的转换规范
- 所有函数必须注明输入/输出物理量与单位
- 禁止静默广播导致形状错误
- 优先向量化实现，但不得牺牲可审查性
- 近似公式与精确公式必须明确区分
- 公式文档可能调整，claude应具备调整公式文档后相应调整代码的能力。
- 参考代码位于reference/文件夹下

## D. 测试规则

- 每个模型至少有：维度检查、极限情况检查、与已知公式/文献结果比对
- 每次修改后优先跑对应模块测试，而不是全项目盲跑
- 如果某个公式尚未确认，先保留为 experimental，而不是伪装成 final

## E. 对 Claude 的行为约束
- 对于任意模块的改动，先给出修改计划，等用户确认后再实施；
- 修改时说明会影响哪些模块；
- 不要大规模重构无关代码；
- 新增代码必须带 docstring、类型标注、最小示例或测试；

## F. 项目架构与开发路线

详细架构定义见：.claude/architecture.md

### a) 目录结构
```
src/qkd_sim/
├── config/          # FiberConfig, WDMConfig, SimulationConfig (YAML+dataclass)
│   └── defaults/    # fiber_smf.yaml, wdm_50ghz.yaml
├── physical/
│   ├── fiber.py     # Fiber类：波长相关loss/dispersion/raman系数
│   ├── signal.py    # SpectrumType枚举, WDMChannel, WDMGrid, G_TX构建
│   ├── noise/       # NoiseSolver ABC, DiscreteSPRSSolver, DiscreteFWMSolver
│   │   ├── base.py          # NoiseSolver ABC
│   │   ├── raman_data.py    # GNPY 92点拉曼系数(内嵌) + get_raman_gain()
│   │   ├── sprs_solver.py   # DiscreteSPRSSolver
│   │   ├── fwm_solver.py    # DiscreteFWMSolver
│   │   └── dispatcher.py    # compute_noise()统一入口
│   ├── spectrum.py  # 噪声/信号功率谱绘图; make_noise_figures()生成8张标准图
│   └── metrics.py   # SNR, OSNR, BER（待实现）
├── network/
│   ├── topology.py, routing.py, traffic.py, resource.py
│   ├── algorithm/   # AllocatorBase ABC, heuristic, ilp, rl_env
│   ├── simulation.py
│   └── evaluator.py
└── utils/units.py   # 单位转换(dB/km→1/m等)
```

### b) 关键设计决策
- 配置管理：YAML文件(常用单位) + dataclass(__post_init__转SI)
- 物理常数：直接 `from scipy.constants import h, k, c`，不设constants.py
- C+L波段：当前仅实现C波段，频率网格不硬编码范围，Fiber支持波长依赖参数
- 拉曼系数：GNPY 92点表内嵌于noise/raman_data.py
- 广播检查：各函数内inline assert，不设独立validation模块
- 噪声求解器：NoiseSolver ABC统一接口，每个solver内部处理离散/连续两条路径
- 网络层算法：AllocatorBase ABC返回(path, wavelength)，仿真循环管理ResourceMap

### c) 开发顺序
1. ~~基础框架(units, config, fiber, signal)~~ ✅ 已完成，29个测试通过
2. ~~离散噪声求解器(SpRS, FWM)~~ ✅ 已完成，53个测试全部通过
3. ~~离散噪声功率谱绘图整理~~ ✅ 一次性计算全部量子信道噪声，信号谱量子信道改竖线，CSV导出
4. ~~连续噪声求解器 + 离散/连续交叉验证~~ ✅ 已完成，59个测试全部通过
   - 新增 `compute_forward_conti()` / `compute_backward_conti()` 方法
   - `compute_noise(continuous=True)` 统一入口
   - 单频极限交叉验证误差 ≤ 5%
   - `tests/test_noise_continuous.py` 6个测试
5. ~~连续模型谱绘图（矩形/升余弦/OSA） + 信号对比图~~ ✅ 已完成
   - `normalize_psd_to_power()` 保证所有连续模型 sum(G·df) = P0
   - `build_wdm_grid(classical_channel_indices=...)` 支持显式指定经典信道
   - Scenario 1: 2×2 多模型噪声谱对比图（W/dBm）
   - Scenario 2: 噪声功率 vs 光纤长度 1×3 对比图
   - `scripts/plot_noise_spectrum.py` 重写为 Phase 4 专用脚本
6. GN-Model经典信道噪声(Poggiolini arXiv:1209.0394 Eq.120+123, 参考oopt-gnpy)
7. 网络层基础(拓扑, 路由, 业务, 资源分配)
8. 仿真集成(事件驱动循环, RL环境)

### d) 已确认的公式修正
- FWM连续模型前向系数：γ²/9（与离散一致，非4γ²/9），已修正formulas_fwm.md
- FWM后向F(l)中exp(α₁z)：z为观测位置（非积分变量），P_{b,1}(0)时自动消去
- 推导来源：Gao et al. JLT 2025 (doi:10.1109/JLT.2025.3610854)
- OSA数据格式：CSV(wavelength_nm, frequency_THz, power_dBm)，存于data/osa/
- **单位修正(2025-04)**：units.py中`alpha_dB_km_to_per_m`公式修正为乘以1e-3（原误写1e3），
  同步修正parameters.md，同步修正`alpha_per_m_to_dB_km`反函数
- FWM噪声求解器策略：方案B（等间隔索引算术），O(N_q×N_c²)，N_c上限250无需FFT优化
- 噪声谱绘图语义（离散模型）：
  - CLASSICAL_INDICES 指定经典信道（泵浦），C波段其余信道同时设为量子信道，
    一次 compute_noise 计算全部量子信道噪声向量（shape (N_q,)）
  - signal_spectrum 中量子信道仅绘制位置竖线（axvline），不显示功率
  - 前向噪声→接收端 z=L（Bob）；后向噪声→发射端 z=0（Alice）
  - 物理假设：经典/量子信道同向传播（co-propagating）
  - 连续模型绘图：continuous=True → compute_forward_conti / compute_backward_conti
- C 波段配置：N_ch=80，50 GHz 间隔，f_center=193.4 THz（~191.4–195.4 THz）
- 测试场景预设：1/3/24 经典信道 × 居中/居左分布；outputs/ 按 discrete_N{N_ch}_C{n_classical} 分目录
- 输出格式：PNG（matplotlib）+ CSV（for Origin），noise_spectrum.csv 含全部噪声分量 W/dBm
- 验证脚本：scripts/plot_noise_spectrum.py（Phase 4 重写），支持离散/连续多模型对比
- **Phase 4 多模型对比图**（scripts/plot_noise_spectrum.py）：
  - Scenario 1（phase4_model_comparison_W/dBm.png）：C 波段 4 种信号模型
    （Discrete/Rectangular/Raised Cosine/OS A）的 FWM/SpRS/总噪声谱 + 信号谱，2×2 布局
  - Scenario 2（phase4_noise_vs_length.png）：固定量子信道频率处，噪声功率
    随光纤长度（1–100 km）变化，FWM/SpRS/总噪声，1×3 布局，log scale
  - PSD 归一化：normalize_psd_to_power() 保证所有连续模型 sum(G·df) = P0
  - OSA 模型：模板峰值对齐各信道 f_center，归一化到 P0
  - 升余弦 β=0.2 用于 raised_cosine 模型