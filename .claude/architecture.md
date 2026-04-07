# 项目架构详细定义

## 1. 核心数据结构与接口

### 1.1 配置层 (config/schema.py)

```python
@dataclass
class FiberConfig:
    """YAML用常用单位，__post_init__转为SI。变量名对应parameters.md。"""
    alpha_dB_per_km: float    # → alpha [1/m]: alpha = alpha_dB * 1e3 / (10 * log10(e))
    gamma_per_W_km: float     # → gamma [1/(W·m)]: gamma = gamma_per_W_km * 1e-3
    D_ps_nm_km: float         # → D_c [s/m²]: D_c = D * 1e-6
    D_slope_ps_nm2_km: float  # → D_slope [s/m³]: D_slope = S * 1e3
    L_km: float               # → L [m]: L = L_km * 1e3
    A_eff: float              # m² (直接SI)
    rayleigh_coeff: float     # 1/m³ (S·alpha_R)
    T_kelvin: float = 300.0

@dataclass
class WDMConfig:
    f_center: float           # Hz (如 193.5e12)
    N_ch: int                 # 信道数
    channel_spacing: float    # Hz (如 50e9)
    B_s: float                # Hz 信号带宽
    P0: float                 # W 单信道发射功率
    beta_rolloff: float = 0.0 # 升余弦滚降因子
    quantum_channel_indices: list[int] = field(default_factory=list)

@dataclass
class SimulationConfig:
    fiber: FiberConfig
    wdm: WDMConfig
    model_type: str           # "discrete" | "continuous"
    spectrum_shape: str       # "rect" | "raised_cosine" | "osa"
    f_grid_resolution: float = 0.1e9  # Hz
```

### 1.2 信号层 (physical/signal.py)

```python
class SpectrumType(Enum):
    SINGLE_FREQ = "single_freq"
    RECTANGULAR = "rectangular"
    RAISED_COSINE = "raised_cosine"
    OSA_SAMPLED = "osa_sampled"

@dataclass
class WDMChannel:
    f_center: float           # Hz
    power: float              # W
    channel_type: str         # "classical" | "quantum"
    spectrum_type: SpectrumType
    B_s: float                # Hz
    beta_rolloff: float = 0.0
    def get_psd(self, f_grid: np.ndarray) -> np.ndarray: ...

@dataclass
class WDMGrid:
    channels: list[WDMChannel]
    f_grid: np.ndarray | None
    def get_classical_channels(self) -> list[WDMChannel]: ...
    def get_quantum_channels(self) -> list[WDMChannel]: ...
    def get_total_psd(self, f_grid: np.ndarray) -> np.ndarray: ...
```

WDM频率网格生成：`f_channels = f_center + np.arange(-(N_ch-1)/2, (N_ch+1)/2) * g`

### 1.3 Fiber类 (physical/fiber.py)

```python
class Fiber:
    def __init__(self, config: FiberConfig): ...
    def get_loss_at_freq(self, freq: np.ndarray) -> np.ndarray: ...      # 1/m
    def get_dispersion_at_freq(self, freq: np.ndarray) -> np.ndarray: ... # s/m²
    def get_raman_gain(self, pump_freq, signal_freq) -> np.ndarray: ...   # 1/(W·m)
    def get_phase_mismatch(self, fi, fj, fk, f_target) -> np.ndarray: ... # rad/m
```

### 1.4 噪声求解器 (physical/noise/)

```python
# base.py
@dataclass
class NoiseResult:
    P_fwd: np.ndarray | None  # (N_ch,) W 或 (N_f,) W/Hz
    P_bwd: np.ndarray | None
    f_axis: np.ndarray
    model_type: str           # "discrete" | "continuous"
    noise_type: str           # "fwm" | "sprs" | "gn_model"
    unit: str                 # "W" | "W/Hz"
    def total_noise(self) -> np.ndarray: ...
    def to_power_in_band(self, f_center: float, bandwidth: float) -> float: ...

class NoiseSolver(ABC):
    def __init__(self, fiber: Fiber): ...
    def compute_forward(self, wdm: WDMGrid, z: float) -> NoiseResult: ...
    def compute_backward(self, wdm: WDMGrid, z: float) -> NoiseResult: ...
    def compute(self, wdm: WDMGrid) -> NoiseResult: ...

# dispatcher.py
def compute_noise(fiber, wdm, enable_fwm=True, enable_raman=True,
                  enable_gn=False) -> dict[str, NoiseResult]: ...
```

### 1.5 网络层 (network/)

```python
# resource.py
class ResourceMap:
    data: np.ndarray  # (N, N, W) int8, 0=不可用/1=空闲/2=占用/3=量子
    def is_available(self, path, wavelength) -> bool: ...
    def allocate(self, path, wavelength) -> None: ...
    def release(self, path, wavelength) -> None: ...

# algorithm/base.py
class AllocatorBase(ABC):
    def allocate(self, event, resource_map, candidate_paths,
                 available_wavelengths) -> tuple[list[int]|None, int|None]: ...
```

## 2. 关键实现注意事项

| 事项 | 说明 |
|------|------|
| 变量命名 | 严格对应 parameters.md "代码变量名"列 |
| FWM系数 | 离散和连续模型前向系数均为 gamma**2/9（已确认） |
| SpRS alpha₁=alpha₂ | `np.where(abs(alpha_diff)<1e-12, L_formula, general)` |
| FWM后向F(l) | exp(alpha₁*z)中z为观测位置，P_{b,1}(0)时自动消去（已确认，Gao et al. JLT 2025） |
| 拉曼系数插值 | GNPY 92点表内嵌于raman_data.py，1D插值(频偏→g_R)，含频率归一化+面积修正 |
| C+L预留 | 频率网格不硬编码C波段(191.7-196.0 THz) |
| 频率/波长域 | Fiber类公开接口统一使用频率(Hz)，内部按需转波长 λ=c/f |
| 量子信道数据流 | WDMConfig.quantum_channel_indices → build_wdm_grid()设置channel_type → NoiseSolver只用classical信道作泵浦 |
| OSA数据 | CSV格式: wavelength_nm, frequency_THz, power_dBm; 需dBm→W/Hz转换(除以RBW) |
| 事件队列 | heapq替代list.sort() |

## 3. 频谱图清单（Phase 4）

每图包含W和dBm两个版本：
1. 信号模型对比图：离散谱 vs 矩形谱 vs 升余弦谱 vs OSA实测谱
2. 量子信道FWM噪声谱
3. 量子信道SpRS噪声谱
4. 量子信道FWM+SpRS合并噪声谱
5. 经典信道非线性噪声谱（Phase 5后补充）
6. 全局叠加图
7. 离散vs连续对比图（FWM、SpRS、总噪声各一组）

## 4. 依赖项

核心：numpy, scipy, pyyaml, matplotlib, networkx
可选：gymnasium (RL), pulp (ILP)
开发：pytest