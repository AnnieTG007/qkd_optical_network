"""噪声求解器抽象基类 (ABC)。

所有噪声求解器（SpRS、FWM、GN-Model）继承 NoiseSolver，
实现 compute_forward / compute_backward 接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qkd_sim.physical.fiber import Fiber
from qkd_sim.physical.signal import WDMGrid


class NoiseSolver(ABC):
    """噪声求解器抽象基类。

    子类须实现 compute_forward 和 compute_backward，
    计算每个量子信道在光纤末端/始端处的噪声功率。

    接口约定
    --------
    - 输入：Fiber（光纤物理参数）+ WDMGrid（信道布局与功率）
    - 输出：shape (N_q,) 的 ndarray，单位 W
    - N_q 为 wdm_grid 中量子信道数量
    - z=L（光纤长度）为默认评估位置；子类可根据需要扩展其他位置

    Notes
    -----
    ABC 是 `collections.abc`（Abstract Base Class）的别名，是 Python 标准库的
    抽象基类机制：
      - 用 ``@abstractmethod`` 装饰的方法子类必须实现，否则实例化时报 TypeError
      - 抽象基类本身不能直接实例化
      - 类似于 Java 的 interface 或 C++ 的纯虚基类
    """

    @abstractmethod
    def compute_forward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算每个量子信道在光纤接收端（z=L）的前向噪声功率。

        Parameters
        ----------
        fiber : Fiber
            光纤物理参数
        wdm_grid : WDMGrid
            WDM 信道网格（含经典/量子信道频率与功率）

        Returns
        -------
        ndarray, shape (N_q,)
            各量子信道前向噪声功率 [W]
        """

    @abstractmethod
    def compute_backward(self, fiber: Fiber, wdm_grid: WDMGrid) -> np.ndarray:
        """计算每个量子信道在光纤发射端（z=0）的后向噪声功率。

        Parameters
        ----------
        fiber : Fiber
            光纤物理参数
        wdm_grid : WDMGrid
            WDM 信道网格（含经典/量子信道频率与功率）

        Returns
        -------
        ndarray, shape (N_q,)
            各量子信道后向噪声功率 [W]
        """