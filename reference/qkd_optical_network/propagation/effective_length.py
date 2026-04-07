"""
有效长度计算

.. deprecated::
    此类已废弃。请直接使用 :meth:`physics.fiber.Fiber.get_effective_length`
"""

import warnings
from typing import Optional
import numpy as np

from physics.fiber import Fiber


class EffectiveLengthCalculator:
    """
    有效长度计算器

    .. deprecated::
        此类已废弃。请使用 :meth:`physics.fiber.Fiber.get_effective_length`

    有效长度定义 [公式 (11)]：
    $$
    L_{eff} = \\frac{1 - e^{-\\alpha L}}{\\alpha}
    $$

    Attributes
    ----------
    fiber : Fiber
        光纤对象
    """

    def __init__(self, fiber: Fiber):
        """
        初始化有效长度计算器

        Parameters
        ----------
        fiber : Fiber
            光纤对象
        """
        warnings.warn(
            "EffectiveLengthCalculator is deprecated. "
            "Use Fiber.get_effective_length(freq) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.fiber = fiber

    def compute(
        self,
        freq: Optional[float] = None,
        use_wavelength_dependent: bool = False
    ) -> float:
        """
        计算有效长度

        Parameters
        ----------
        freq : float, optional
            频率 [Hz]。如果为 None，使用 193.4 THz。
        use_wavelength_dependent : bool, optional
            如果 True 且定义了波长相关衰减，则使用插值计算。
            如果 False，使用常数衰减。

        Returns
        -------
        L_eff : float
            有效长度 [m]

        References
        ----------
        - FORMULAS_REVISION.md 公式 (11)
        """
        if freq is None:
            freq = 193.4e12  # 默认 C 波段中心频率

        if use_wavelength_dependent and self.fiber.wavelength_dependent_loss is not None:
            alpha = self.fiber.get_loss_at_freq(freq)
        else:
            alpha = self.fiber.loss

        # 避免除零
        if alpha < 1e-15:
            return self.fiber.length

        return (1 - np.exp(-alpha * self.fiber.length)) / alpha

    def compute_array(self, freq_array: np.ndarray) -> np.ndarray:
        """
        计算多个频率处的有效长度数组

        Parameters
        ----------
        freq_array : np.ndarray
            频率数组 [Hz]

        Returns
        -------
        L_eff_array : np.ndarray
            有效长度数组 [m]，与 freq_array 同形状
        """
        alpha_array = self.fiber.get_loss_array(freq_array)

        # 避免除零
        L_eff_array = np.where(
            alpha_array < 1e-15,
            self.fiber.length,
            (1 - np.exp(-alpha_array * self.fiber.length)) / alpha_array
        )

        return L_eff_array
