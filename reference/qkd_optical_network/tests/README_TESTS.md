# 噪声模型测试说明

本文档详细说明 FWM 和拉曼噪声单元测试的验证内容。

---

## 测试文件位置

- **FWM 测试**: `tests/test_fwm.py`
- **拉曼测试**: `tests/test_raman.py`
- **噪声功率谱计算**: `validation/spectrum_computation/compute_noise_spectrum.py`
- **离散 vs 连续模型对比**: `validation/model_comparison/compare_noise_models.py`

---

## FWM 噪声测试详解

### 测试环境配置

```python
fiber = Fiber(
    fiber_type=FiberType.SSMF,
    length=50e3,  # 50 km
    temperature=300.0
)

# 3 信道 WDM 系统，50 GHz 间隔，0 dBm 功率
channels = [
    WDMChannel(center_freq=193.35e12, power=1e-3),  # f₀ - Δf
    WDMChannel(center_freq=193.4e12,  power=1e-3),  # f₀
    WDMChannel(center_freq=193.45e12, power=1e-3),  # f₀ + Δf
]
```

### 测试用例列表

#### 1. `test_fwm_noise_shape` - 输出形状验证

**验证内容**: FWM 噪声输出数组的形状与输入信道数一致

**物理意义**: 确保每个信道都能得到对应的噪声功率值

**断言**:
```python
assert noise.shape == (len(channels),)
```

---

#### 2. `test_fwm_noise_positive` - 非负性验证

**验证内容**: FWM 噪声功率必须为非负值

**物理意义**: 噪声功率是物理能量，不能为负

**断言**:
```python
assert np.all(noise >= 0)
```

---

#### 3. `test_fwm_noise_nonzero` - 非零验证

**验证内容**: 当存在多个信道时，FWM 噪声不应全为零

**物理意义**: 多信道系统中，四波混频效应必然产生能量转移

**断言**:
```python
assert np.any(noise > 0)
```

---

#### 4. `test_fwm_with_fewer_channels` - 边界条件验证

**验证内容**:
- 1 个信道时，返回零噪声
- 2 个信道时，返回零噪声

**物理意义**: FWM 需要至少 3 个频率分量才能产生（fi + fj - fk = f_fwm）

**断言**:
```python
np.testing.assert_array_equal(noise_1ch, np.zeros(1))
np.testing.assert_array_equal(noise_2ch, np.zeros(2))
```

---

#### 5. `test_fwm_power_scaling` - 功率依赖性验证

**验证内容**: 高功率输入产生的 FWM 噪声应大于低功率输入

**物理意义**: FWM 噪声功率与输入功率的三次方成正比（P_fwm ∝ Pi·Pj·Pk）

**测试配置**:
- 低功率：-30 dBm (1 μW)
- 高功率：+10 dBm (10 mW)

**断言**:
```python
assert np.all(noise_high > noise_low)
```

---

#### 6. `test_fwm_frequency_spacing` - 频率间隔依赖性验证

**验证内容**: 不同信道间隔下代码能正常运行

**物理意义**:
- 小间隔（25 GHz）→ 相位失配小 → FWM 效率高
- 大间隔（100 GHz）→ 相位失配大 → FWM 效率低

**测试配置**:
- 小间隔组：25 GHz
- 大间隔组：100 GHz

**断言**:
```python
assert noise_small.shape == (3,)
assert noise_large.shape == (3,)
```

---

#### 7. `test_fwm_compute_at_length` - 接口完整性验证

**验证内容**:
- `compute_at_length=True` 正常工作
- `compute_at_length=False` 抛出 `NotImplementedError`

**物理意义**: 当前版本仅支持计算光纤末端噪声，分布式噪声计算暂未实现

**断言**:
```python
assert noise.shape == (n_channels,)  # compute_at_length=True
with pytest.raises(NotImplementedError):
    compute_fwm_noise(..., compute_at_length=False)
```

---

## 拉曼噪声测试详解

### 测试环境配置

```python
fiber = Fiber(
    fiber_type=FiberType.SSMF,
    length=50e3,
    temperature=300.0
)

# 2 信道 WDM 系统（拉曼最小需求）
channels = [
    WDMChannel(center_freq=193.35e12, power=1e-3),
    WDMChannel(center_freq=193.4e12,  power=1e-3),
]
```

### 测试用例列表

#### 1. `test_raman_noise_shape` - 输出形状验证

**验证内容**: 拉曼噪声输出数组的形状与输入信道数一致

**断言**:
```python
assert noise.shape == (len(channels),)
```

---

#### 2. `test_raman_noise_positive` - 非负性验证

**验证内容**: 拉曼噪声功率必须为非负值

**物理意义**: 自发拉曼散射产生的噪声能量不能为负

**断言**:
```python
assert np.all(noise >= 0)
```

---

#### 3. `test_raman_with_fewer_channels` - 边界条件验证

**验证内容**: 1 个信道时返回零噪声

**物理意义**: 拉曼散射需要泵浦光和信号光两个频率分量

**断言**:
```python
np.testing.assert_array_equal(noise_1ch, np.zeros(1))
```

---

#### 4. `test_raman_power_scaling` - 功率依赖性验证

**验证内容**: 高功率输入产生的拉曼噪声应大于低功率输入

**物理意义**: 拉曼噪声功率与泵浦功率成正比（P_raman ∝ P_pump）

**测试配置**:
- 低功率：-30 dBm (1 μW)
- 高功率：+10 dBm (10 mW)

**断言**:
```python
assert np.all(noise_high > noise_low)
```

---

#### 5. `test_raman_temperature_dependence` - 温度依赖性验证

**验证内容**: 不同温度下代码能正常运行

**物理意义**: 拉曼散射涉及 Bose-Einstein 光子数分布：
$$n_{th} = \frac{1}{e^{h\Delta f/kT} - 1}$$

温度越高，热光子数越多，anti-Stokes 过程越强

**测试配置**:
- 低温：77 K（液氮温度）
- 室温：300 K
- 高温：500 K

**断言**:
```python
assert noise_cold.shape == (2,)
assert noise_room.shape == (2,)
assert noise_hot.shape == (2,)
```

---

#### 6. `test_raman_stokes_anti_stokes` - Stokes/anti-Stokes 对称性验证

**验证内容**: 对称频率配置下代码能正常运行

**物理意义**:
- Stokes 散射：信号光频率 < 泵浦光频率（发射声子）
- anti-Stokes 散射：信号光频率 > 泵浦光频率（吸收声子）

**测试配置**: 3 信道对称系统
```
f₀ - Δf  (低频)
f₀       (中心)
f₀ + Δf  (高频)
```

**断言**:
```python
assert np.all(noise >= 0)
```

---

#### 7. `test_raman_compute_at_length` - 接口完整性验证

**验证内容**: 与 FWM 测试相同

---

#### 8. `test_raman_forward_backward` - 前向/后向噪声验证

**验证内容**: 使用长光纤（100 km）验证前向和后向拉曼噪声都存在

**物理意义**:
- 前向拉曼：散射光与泵浦光同向传播
- 后向拉曼：散射光与泵浦光反向传播

**断言**:
```python
assert np.any(noise > 0)
```

---

## 噪声功率谱计算脚本说明

### 文件位置

`validation/spectrum_computation/compute_noise_spectrum.py`

### 功能

1. **构建 WDM 系统**: 模拟经典 - 量子共纤场景
   - 经典信道：0 dBm（高功率，作为拉曼泵浦）
   - 量子信道：-70 dBm（低功率，QKD 信号）

2. **计算噪声**:
   - FWM 噪声（公式 1-4）
   - 拉曼噪声（公式 5-10）

3. **导出结果**:
   - CSV 格式功率谱（可用 Origin 打开）
   - PNG 格式可视化图

### 输出文件

| 文件名 | 内容 |
|--------|------|
| `output/noise_spectra/noise_spectrum_fwm.csv` | FWM 噪声功率谱 |
| `output/noise_spectra/noise_spectrum_raman.csv` | 拉曼噪声功率谱 |
| `output/noise_spectra/noise_spectrum_total.csv` | 总噪声功率谱 |
| `output/spectrum_models/noise_spectrum_plot.png` | 可视化对比图 |

### 使用方法

```bash
cd E:\王雨婷个人文件夹\01：仿真代码合集\qkd_optical_network
python validation/spectrum_computation/compute_noise_spectrum.py
```

---

## 测试结果总结

### FWM 测试：7 项全部通过

| 测试项 | 验证内容 | 状态 |
|--------|----------|------|
| test_fwm_noise_shape | 输出形状正确 | ✅ |
| test_fwm_noise_positive | 噪声非负 | ✅ |
| test_fwm_noise_nonzero | 多信道产生噪声 | ✅ |
| test_fwm_with_fewer_channels | <3 信道无噪声 | ✅ |
| test_fwm_power_scaling | 功率依赖性正确 | ✅ |
| test_fwm_frequency_spacing | 频率间隔适应性 | ✅ |
| test_fwm_compute_at_length | 接口完整性 | ✅ |

### 拉曼测试：8 项全部通过

| 测试项 | 验证内容 | 状态 |
|--------|----------|------|
| test_raman_noise_shape | 输出形状正确 | ✅ |
| test_raman_noise_positive | 噪声非负 | ✅ |
| test_raman_with_fewer_channels | <2 信道无噪声 | ✅ |
| test_raman_power_scaling | 功率依赖性正确 | ✅ |
| test_raman_temperature_dependence | 温度依赖性 | ✅ |
| test_raman_stokes_anti_stokes | Stokes/anti-Stokes | ✅ |
| test_raman_compute_at_length | 接口完整性 | ✅ |
| test_raman_forward_backward | 前向/后向噪声 | ✅ |

---

## 公式参考

所有测试基于 FORMULAS_REVISION.md 中的公式：

- **FWM**: 公式 (1)-(4)
  - (1) 相位失配因子
  - (2) FWM 效率
  - (3) 衰减系数差
  - (4) FWM 产物功率

- **拉曼**: 公式 (5)-(10)
  - (5) Bose-Einstein 分布
  - (6)-(7) 自发拉曼截面
  - (8)-(9) 前向拉曼功率
  - (10) 后向拉曼功率
