"""
光纤标准参数定义

包含 SMF（标准单模光纤）和 HCF（空芯光纤）的标准参数。
所有参数使用国际单位制（SI）。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


class FiberType(Enum):
    """光纤类型枚举"""
    SSMF = "SSMF"  # 标准单模光纤
    HCF = "HCF"  # 空芯光纤


@dataclass
class FiberParameters:
    """
    光纤参数数据类

    所有参数使用国际单位制（SI）：
    - 长度：m
    - 频率：Hz
    - 功率：W
    - 衰减：m⁻¹
    """
    fiber_type: FiberType

    # 基本参数
    loss: float  # 光纤衰减系数 [m⁻¹]
    nonlinear_coff: float  # 非线性系数 γ [W⁻¹·m⁻¹]
    cd_coff: float  # 色散系数 D [s/m²]
    cd_slope: float  # 色散斜率 S [s/m³]
    effective_area: float  # 有效模场面积 [m²]
    refractive_index: float  # 折射率 n

    # 拉曼散射相关
    raman_data: Optional[dict] = None  # 拉曼增益系数表

    # 瑞利散射相关（HCF 专用，SMF 可设为 None）
    rayleigh_loss: Optional[float] = None  # 瑞利散射衰减 [m⁻¹]
    recapture_factor_rayleigh: Optional[float] = None  # 后向瑞利散射捕获因子

    # 可选参数
    length: float = 50e3  # 默认光纤长度 [m]
    temperature: float = 300.0  # 工作温度 [K]


# ========== GNpy 拉曼系数表（SSMF） ==========
# 来源：GNpy 开源项目
# 用于查表法计算拉曼增益系数 gR(Δf)

GNPY_RAMAN_COEFFICIENT = {
    # SSMF Raman coefficient profile in terms of mode intensity (g0 * A_eff overlap)
    # https://gnpy.readthedocs.io/en/master/model.html
    'gamma_raman': np.array(
        [0.0, 8.524419934705497e-16, 2.643567866245371e-15, 4.410548410941305e-15, 6.153422961291078e-15,
         7.484924703044943e-15, 8.452060808349209e-15, 9.101549322698156e-15, 9.57837595158966e-15,
         1.0008642675474562e-14, 1.0865773569905647e-14, 1.1300776305865833e-14, 1.2143238647099625e-14,
         1.3231065750676068e-14, 1.4624900971525384e-14, 1.6013330554840492e-14, 1.7458119359310242e-14,
         1.9320241330434762e-14, 2.1720395392873534e-14, 2.4137337406734775e-14, 2.628163218460466e-14,
         2.8041019963285974e-14, 2.9723155447089933e-14, 3.129353531005888e-14, 3.251796163324624e-14,
         3.3198839487612773e-14, 3.329527690685666e-14, 3.313155691238456e-14, 3.289013852154548e-14,
         3.2458917188506916e-14, 3.060684277937575e-14, 3.2660349473783173e-14, 2.957419109657689e-14,
         2.518894321396672e-14, 1.734560485857344e-14, 9.902860761605233e-15, 7.219176385099358e-15,
         6.079565990401311e-15, 5.828373065963427e-15, 7.20580801091692e-15, 7.561924351387493e-15,
         7.621152352332206e-15, 6.8859886780643254e-15, 5.629181047471162e-15, 3.679727598966185e-15,
         2.7555869742500355e-15, 2.4810133942597675e-15, 2.2160080532403624e-15, 2.1440626024765557e-15,
         2.33873070799544e-15, 2.557317929858713e-15, 3.039839048226572e-15, 4.8337165515610065e-15,
         5.4647431818257436e-15, 5.229187813711269e-15, 4.510768525811313e-15, 3.3213473130607794e-15,
         2.2602577027996455e-15, 1.969576495866441e-15, 1.5179853954188527e-15, 1.2953988551200156e-15,
         1.1304672156251838e-15, 9.10004390675213e-16, 8.432919922183503e-16, 7.849224069008326e-16,
         7.827568196032024e-16, 9.000514440646232e-16, 1.3025926460013665e-15, 1.5444108938497558e-15,
         1.8795594063060786e-15, 1.7796130169921014e-15, 1.5938159865046653e-15, 1.1585522355108287e-15,
         8.507044444633358e-16, 7.625404663756823e-16, 8.14510750925789e-16, 9.047944693473188e-16,
         9.636431901702084e-16, 9.298633899602105e-16, 8.349739503637023e-16, 7.482901278066085e-16,
         6.240794767134268e-16, 5.00652535687506e-16, 3.553373263685851e-16, 2.0344217706119682e-16,
         1.4267522642294203e-16, 8.980016576743517e-17, 2.9829068181832594e-17, 1.4861959129014824e-17,
         7.404482113326137e-18]
    ),  # m/W
    # SSMF Raman coefficient profile (g0, 拉曼增益系数)
    'g0': np.array(
        [0.00000000e+00, 1.12351610e-05, 3.47838074e-05, 5.79356636e-05, 8.06921680e-05, 9.79845709e-05,
         1.10454361e-04, 1.18735302e-04, 1.24736889e-04, 1.30110053e-04, 1.41001273e-04, 1.46383247e-04,
         1.57011792e-04, 1.70765865e-04, 1.88408911e-04, 2.05914127e-04, 2.24074028e-04, 2.47508283e-04,
         2.77729174e-04, 3.08044243e-04, 3.34764439e-04, 3.56481704e-04, 3.77127256e-04, 3.96269124e-04,
         4.10955175e-04, 4.18718761e-04, 4.19511263e-04, 4.17025384e-04, 4.13565369e-04, 4.07726048e-04,
         3.83671291e-04, 4.08564283e-04, 3.69571936e-04, 3.14442090e-04, 2.16074535e-04, 1.23097823e-04,
         8.95457457e-05, 7.52470400e-05, 7.19806145e-05, 8.87961158e-05, 9.30812065e-05, 9.37058268e-05,
         8.45719619e-05, 6.90585286e-05, 4.50407159e-05, 3.36521245e-05, 3.02292475e-05, 2.69376939e-05,
         2.60020897e-05, 2.82958958e-05, 3.08667558e-05, 3.66024657e-05, 5.80610307e-05, 6.54797937e-05,
         6.25022715e-05, 5.37806442e-05, 3.94996621e-05, 2.68120644e-05, 2.33038554e-05, 1.79140757e-05,
         1.52472424e-05, 1.32707565e-05, 1.06541760e-05, 9.84649374e-06, 9.13999627e-06, 9.08971012e-06,
         1.04227525e-05, 1.50419271e-05, 1.77838232e-05, 2.15810815e-05, 2.03744008e-05, 1.81939341e-05,
         1.31862121e-05, 9.65352116e-06, 8.62698322e-06, 9.18688016e-06, 1.01737784e-05, 1.08017817e-05,
         1.03903588e-05, 9.30040333e-06, 8.30809173e-06, 6.90650401e-06, 5.52238029e-06, 3.90648708e-06,
         2.22908227e-06, 1.55796177e-06, 9.77218716e-07, 3.23477236e-07, 1.60602454e-07, 7.97306386e-08]
    ),  # [1/(W·m)]
    # 频率偏移（非均匀间隔，用于精确捕捉拉曼峰形状）
    'frequency_offset': np.array([
        0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11.,
        11.5, 12., 12.5, 12.75, 13., 13.25, 13.5, 14., 14.5, 14.75, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.25,
        18.5, 18.75, 19., 19.5, 20., 20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.5, 26., 26.5, 27.,
        27.5, 28., 28.5, 29., 29.5, 30., 30.5, 31., 31.5, 32., 32.5, 33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5,
        37., 37.5, 38., 38.5, 39., 39.5, 40., 40.5, 41., 41.5, 42.
    ]) * 1e12,  # [Hz]
    # 参考频率（1454 nm）
    'reference_frequency': 206.184634112792e12,  # [Hz]
    # 参考有效面积（@1454 nm）
    'reference_effective_area': 75.74659443542413e-12,  # [m²]
}


# ========== 标准单模光纤 (SMF) 参数 ==========
# 来源：综合多个商用 SMF 规格书

SSMF_PARAMETERS = FiberParameters(
    fiber_type=FiberType.SSMF,

    # 基本参数
    loss=0.20 / 4.343 * 1e-3,  # 0.20 dB/km → [m⁻¹]
    nonlinear_coff=1.3e-3,  # 非线性系数 γ [W⁻¹·m⁻¹]
    cd_coff=17e-6,  # 色散系数 D = 17 ps/(nm·km) [s/m²]
    cd_slope=56e3,  # 色散斜率 S = 0.056 ps/(nm²·km) [s/m³]
    effective_area=70e-12,  # 有效模场面积 70 μm² [m²]
    refractive_index=1.45,  # 折射率

    # 瑞利散射相关
    rayleigh_loss=3.2e-5,  # 瑞利散射衰减 [m⁻¹]
    recapture_factor_rayleigh=1.5e-3,  # 后向瑞利散射捕获因子

    # 拉曼数据（使用 GNpy 数据表）
    raman_data=GNPY_RAMAN_COEFFICIENT,

    # 默认参数
    length=50e3,  # 默认长度 50 km [m]
    temperature=300.0,  # 工作温度 300 K
)


# ========== 空芯光纤 (HCF) 参数 ==========
# 来源：
# - 衰减：Hollow Core DNANF Optical Fiber with <0.11 dB/km Loss
# - 非线性系数：ACP2024 会议鹏程实验室报告
# - 瑞利散射：Loss in Hollow-Core Optical Fibers Mechanisms, Scaling Rules, and Limits
# - 色散：Distribution of Telecom Entangled Photons Through 7.7 km Anti-resonant Hollow-Core Fiber
# - 色散斜率：Stable Optical Frequency Comb Enabled by Hollow-Core Fibers

HCF_PARAMETERS = FiberParameters(
    fiber_type=FiberType.HCF,

    # 基本参数
    loss=0.11 / 4.343 * 1e-3,  # 0.11 dB/km → [m⁻¹]
    nonlinear_coff=5.0e-7,  # 非线性系数 γ [W⁻¹·m⁻¹]
    cd_coff=2.0e-6,  # 色散系数 D = 2.0 ps/(nm·km) [s/m²]
    cd_slope=4e-3,  # 色散斜率 S = 4 fs/(km·nm²) [s/m³]
    effective_area=75.74659443542413e-12,  # 有效模场面积 [m²]（与 GNpy 参考一致）
    refractive_index=1.45,  # 折射率（近似）

    # 瑞利散射相关
    rayleigh_loss=8.0e-4 / 4.343 * 1e-3,  # 8e-4 dB/km → [m⁻¹]
    recapture_factor_rayleigh=5.875e-22,  # 后向瑞利散射捕获因子

    # 拉曼数据（使用 GNpy 数据表，暂用 SSMF 数据）
    # TODO: 后续替换为 HCF 实测拉曼数据
    raman_data=GNPY_RAMAN_COEFFICIENT,

    # 默认参数
    length=50e3,  # 默认长度 50 km [m]
    temperature=300.0,  # 工作温度 300 K
)


def get_fiber_parameters(
    fiber_type: FiberType = FiberType.SSMF,
    custom_params: Optional[dict] = None
) -> FiberParameters:
    """
    获取光纤参数

    Parameters
    ----------
    fiber_type : FiberType
        光纤类型（SSMF 或 HCF）
    custom_params : dict, optional
        自定义参数，用于覆盖默认值

    Returns
    -------
    FiberParameters
        光纤参数对象

    Examples
    --------
    >>> params = get_fiber_parameters(FiberType.SSMF)
    >>> params.loss  # 获取衰减系数
    >>> params = get_fiber_parameters(FiberType.HCF, {'length': 100e3})  # 自定义长度
    """
    # 获取默认参数
    if fiber_type == FiberType.SSMF:
        params = SSMF_PARAMETERS
    elif fiber_type == FiberType.HCF:
        params = HCF_PARAMETERS
    else:
        raise ValueError(f"Unknown fiber type: {fiber_type}")

    # 覆盖自定义参数
    if custom_params:
        # 使用 dataclass 的 replace 方法（或直接修改属性）
        for key, value in custom_params.items():
            if hasattr(params, key):
                setattr(params, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    return params
