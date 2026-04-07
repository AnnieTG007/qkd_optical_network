import numpy as np
import pandas as pd
import math

# ————全局常数区————
# STD_INDEX对应，不知道为什么在以前的代码里标的300
STD_INDEX = 200
STD_FREQ = 193.4e12
INV_FREQ = 25e9
PLANCK_CONSTANT = 6.62607015 * 10 ** (-34)
C = 299792458


class Fiber:
    def __init__(self, fiber_type='HCF', qkd_protocol='BB84'):
        # ————光纤参数————
        if fiber_type == 'HCF':
            # 这个建模可能不严谨的地方在于数据来源不唯一，如果出了比较全的商用数据可以替换
            # 光纤衰减(单位：m^-1)
            # 来源：Hollow Core DNANF Optical Fiber with <0.11 dB/km Loss
            self.loss = 0.11 / 4.343 * 1e-3
            # 非线性系数(单位：W^-1*m^-1)
            # 来源：ACP2024会议鹏程实验室报告
            self.nonlinear_coff = 5.0e-7
            # self.nonlinear_coff = 2 * np.pi * self.n2 / (self.wavelength * self.A)

            # ————瑞利散射相关————
            # 瑞利散射的衰减（单位：m^-1）
            # 来源：Loss in Hollow-Core Optical Fibers Mechanisms, Scaling Rules, and Limits
            # 原文所给的数值是8 * 10 ^ (-4) dB/km，需要统一为国际单位制
            self.rayleigh_loss = 8.0e-4 / 4.343 * 1e-3
            # 后向瑞利散射捕获因子
            # 来源：没查到，S * alpha_R = rayleigh_coff（南安普顿的研究仓库？）换算的
            self.recapture_factor_Rayleigh = 5.875 * 1e-22
            # 瑞利散射系数（单位：m^-3）
            # 来源：Quasi Single-Mode Fiber With Record-Low Attenuation of 0.1400 dB/km
            # 原文：Rayleigh scattering coefficient in the hollow regions is 4.7 × 10−3 dB·μm4/km
            self.rayleigh_coff = 4.7 / 4.343 * 1e-30

            # ————色散相关————
            # 色散系数(单位：s/m^2)
            # 来源1：ACP2024会议鹏程实验室报告
            # 来源2：Distribution of Telecom Entangled Photons Through topo 7.7 km Anti-resonant Hollow-Core Fiber
            # 原文所给数值为 2.0 ps / (nm * km)
            self.cd_coff = 2.0e-6
            # 色散斜率(单位：s/m^3)
            # 来源：Stable Optical Frequency Comb Distribution Enabled by Hollow-Core Fibers
            # 原文所给的数值为4 fs/(km * nm ^ 2)，需要统一为国际单位制，其中fs 飞秒 代表1e-15 s
            self.cd_slope = 4e-3

        elif fiber_type == 'SMF':
            self.loss = 0.20 / 4.343 * 1e-3                               # 光纤衰减(单位：m^-1)
            self.nonlinear_coff = 1.3e-3                                  # 非线性系数(单位：W^-1*m^-1)
            self.recapture_factor_Rayleigh = 1.5e-3
            self.rayleigh_loss = 3.2e-5                                   # 瑞利散射的衰减3.2 * 10 ^ (-2) / km，转化为m
            self.cd_coff = 1.7e-5                                         # 色散系数(单位：s/m^2)
            self.cd_slope = 56                                            # 色散斜率(单位：s/m^3)

        self.len = 50e3                                                   # 光纤长度(单位：m)
        self.raman_data = pd.read_excel(
            'raman_cross_section_25GHz.xls',
            sheet_name=fiber_type, header=None)                           # 有效拉曼横截面积(单位：m^-1*m^-1)

        # ————经典光纤通信系统参数————
        self.quantum_receiver_width = 0.8e-9                              # 量子接收机带宽(单位：m)
        self.pump = 193.4e12                                              # 泵浦光源中心频率(单位：Hz)
        self.gate_time = 1000 * 10 ** (-12)                               # 探测时间(单位：s)
        self.detector_eff = 0.1                                           # 探测效率
        self.IL = 8                                                       # 密集波分复用系统插入损耗(单位：dB)

        # ————量子密钥分发参数————
        if qkd_protocol == 'BB84':
            self.qkd_eff = 0.5                                            #

    # ————从非线性噪声产生的角度计算————
    def raman_scatter_forward(self, p_in: float, f_pump: float, freq: float):
        """
        计算前向自发拉曼散射噪声
        :param p_in: 输入功率（单位：W）
        :param f_pump: 泵浦源中心频率（单位：Hz）
        :param freq: 产生噪声的经典信道中心频率（单位：Hz）
        :return: 计算得到的前向自发拉曼散射噪声的值（单位：W）
        """
        # 计算时考虑衰减
        p_out = p_in * np.exp(-self.loss * self.len)

        # 泵浦源波长未必是1550nm，需要计算相对波长
        f_delta = STD_FREQ + (freq - f_pump)

        # 以193.4 THz / 1550nm 为基准，在拉曼散射因子数据表中，寻找相对波长的位置
        index = int(math.floor((STD_FREQ - f_delta) / INV_FREQ)) + STD_INDEX
        raman_cross_section = self.raman_data.at[index, 1] * (f_delta / freq) ** 4
        return p_out * self.len * raman_cross_section * self.quantum_receiver_width

    def raman_scatter_backward(self, p_in: float, f_pump: float, freq: float):
        """
        计算后向自发拉曼散射噪声
        :param p_in: 输入功率（单位：W）
        :param f_pump: 泵浦源中心频率（单位：Hz）
        :param freq: 产生噪声的经典信道中心频率（单位：Hz）
        :return: 计算得到的后向自发拉曼散射噪声的值（单位：W）
        """
        # 计算时考虑衰减
        p_out = p_in * np.exp(-self.loss * self.len)

        # 泵浦源波长未必是1550nm，需要计算相对波长
        f_delta = STD_FREQ + (freq - f_pump)

        # 以193.4 THz / 1550nm 为基准，在拉曼散射因子数据表中，寻找中心频率为freq经典信道的位置
        index = int(round((STD_FREQ - f_delta) / INV_FREQ)) + STD_INDEX
        raman_cross_section = self.raman_data.at[index, 1] * (f_delta / freq) ** 4
        return p_out * math.sinh(self.len * self.loss) / self.loss * raman_cross_section * self.quantum_receiver_width

    def four_wave_mixing_forward(self, pi: float, pj: float, pk: float, fi: float, fj: float, fk: float):
        """
        计算前向四波混频噪声功率
        :param pi: 第一个频率对应的功率
        :param pj: 第二个频率对应的功率
        :param pk: 第三个频率对应的功率
        :param fi: 第一个频率
        :param fj: 第二个频率
        :param fk: 第三个频率
        :return: f_fwm: 四波混频所在频率，p_fwm: 四波混频噪声功率
        """
        # 计算简并因子
        if np.abs(fi - fj) < 10 ** 6:
            d = 3  # fi和fj相等
        else:
            d = 6  # fi和fj不相等

        # 求解四波混频所在频率和波长
        f_fwm = fi + fj - fk
        w_fwm = C / f_fwm

        # 求解相位匹配因子
        beta = (2 * np.pi * w_fwm ** 2 / C * np.abs(fi - fk) * np.abs(fj - fk)
                * (self.cd_coff + self.cd_slope * 0.5 * w_fwm ** 2 / C * (np.abs(fi - fk) + np.abs(fj - fk))))

        # 求解四波混频效率，exp和tamp无物理意义，设定只是为了简化公式输入
        exp = np.exp(-self.loss * self.len)
        temp = (4 * exp * (np.sin(0.5 * beta * self.len) ** 2) / (1 - exp) ** 2)
        eta = self.loss ** 2 / (self.loss ** 2 + beta ** 2) * (1 + temp)

        # 求解非线性系数
        gamma = self.nonlinear_coff

        # 计算四波混频噪声功率
        p_fwm = (eta * (d ** 2) * (gamma ** 2) * pi * pj * pk * exp / (9 * self.loss ** 2) * (1 - exp) ** 2)

        return f_fwm, p_fwm

    def four_wave_mixing_backward(self, pi: float, pj: float, pk: float, fi: float, fj: float, fk: float):
        """
        计算后向四波混频噪声功率
        :param pi: 第一个频率对应的功率
        :param pj: 第二个频率对应的功率
        :param pk: 第三个频率对应的功率
        :param fi: 第一个频率
        :param fj: 第二个频率
        :param fk: 第三个频率
        :return: f_fwm: 四波混频所在频率，p_fwm: 四波混频噪声功率
         """
        # 计算简并因子
        if np.abs(fi - fj) < 10 ** 6:
            d = 3  # fi和fj相等
        else:
            d = 6  # fi和fj不相等

        # 求解四波混频所在频率和波长
        f_fwm = fi + fj - fk
        w_fwm = C / f_fwm

        # 求解相位匹配因子
        beta = (2 * np.pi * w_fwm ** 2 / C * np.abs(fi - fk) * np.abs(fj - fk)
                * (self.cd_coff + self.cd_slope * 0.5 * w_fwm ** 2 / C * (np.abs(fi - fk) + np.abs(fj - fk))))

        # 求解非线性系数
        gamma = self.nonlinear_coff

        # 求解四波混频效率
        # 对距离而言的常系数，积分时可提取至积分号外
        con_coff = (self.recapture_factor_Rayleigh * self.rayleigh_loss / (self.loss ** 2 + beta ** 2) *
                    d ** 2 * gamma ** 2 * pi * pj * pk / 9)

        # 与距离有关的量，需要参与积分，这里直接给出积分结果
        # 积分求解网站： https://mathdf.com/int/cn/
        # 积分式 (1+4*e^(-alphax) * sin^2(beta*x/2)/(1-e^(-alpha*x))^2)*e^(-2*alpha*x)*(1-e^(-alpha*x))^2)
        # exp sin cos是为了简化表达式提取出来的变量
        exp = np.exp(-self.loss * self.len)
        sin = np.sin(beta * self.len)
        cos = np.cos(beta * self.len)
        # div代表分母，为了简化表达式
        div = -1 / (4 * self.loss * beta ** 2 + 36 * self.loss ** 3)
        # e^-4项的系数
        a4 = (beta ** 2 + 9 * self.loss ** 2) * exp ** 4
        # e^-3项的系数
        a3 = (8 * self.loss * beta * sin - 24 * self.loss ** 2 * cos) * exp ** 3
        # e^-2项的系数
        a2 = (2 * beta ** 2 + 18 * self.loss ** 2) * exp ** 2
        # 常数项系数
        a0 = (-3 * beta ** 2 - 3 * self.loss ** 2)
        var_coff = div * (a4 + a3 + a2 + a0)

        # 四波混频噪声功率
        p_fwm = con_coff * var_coff
        return f_fwm, p_fwm

    def calculate_fwm_noise(self, channel_list: list, channel_dir: np.ndarray, channel_power: np.ndarray):
        """
        计算每个信道所受四波混频噪声功率
        :param channel_list: 列表，包含所有信道的中心频率
        :param channel_dir: 一维矩阵，代表对应信道上信号的方向（0前向信号，1后向信号）
        :param channel_power: 一维矩阵，代表所有信道的功率（功率为0代表该信道上没有信号）
        :return: 信道上的噪声功率列表（单位: W）
        """
        # 信道数量
        channel_num = len(channel_list)
        noise_power = np.zeros(channel_num)

        # 计算四波混频噪声功率
        for fi in channel_list:
            index_i = channel_list.index(fi)
            pi = channel_power[index_i]
            # 枚举fi，注意当信道功率为0时代表该信道没有信号
            if pi == 0:
                continue
            for fj in channel_list:
                index_j = channel_list.index(fj)
                pj = channel_power[index_j]
                # 枚举fj，注意当信道功率为0时代表该信道没有信号
                if pj == 0 or index_j < index_i or channel_dir[index_j] != channel_dir[index_i]:
                    continue
                for fk in channel_list:
                    index_k = channel_list.index(fk)
                    pk = channel_power[index_k]
                    # 枚举fk，注意当信道功率为0时代表该信道没有信号
                    if pk == 0 or channel_dir[index_k] != channel_dir[index_i]:
                        continue
                    # 四波混频条件要求fk不能与fi或fj相等
                    # 否则能量由fi和fj交换至fk和f_fwm的过程就没有意义了
                    if fi == fk or fj == fk:
                        continue
                    f_fwm = fi + fj - fk
                    # 如果四波混频产生的信道在信道范围外，则不用考虑
                    if f_fwm not in channel_list:
                        continue
                    if channel_dir[index_i] == 'f':
                        f_fwm, p_fwm = self.four_wave_mixing_forward(pi, pj, pk, fi, fj, fk)
                    else:
                        f_fwm, p_fwm = self.four_wave_mixing_backward(pi, pj, pk, fi, fj, fk)
                    noise_power[channel_list.index(f_fwm)] += p_fwm
        return noise_power

    def calculate_sprs_noise(self, channel_list: list, channel_dir: np.ndarray, channel_power: np.ndarray):
        """
        计算每个信道所受自发拉曼散射噪声功率
        :param channel_list: 列表，包含所有信道的中心频率
        :param channel_dir: 一维矩阵，代表对应信道上信号的方向（0前向信号，1后向信号）
        :param channel_power: 一维矩阵，代表所有信道的功率（功率为0代表该信道上没有信号）
        :return: 信道上的噪声功率列表（单位: W）
        """
        # 信道数量
        channel_num = len(channel_list)
        noise_power = np.zeros(channel_num)

        # 计算自发拉曼散射噪声功率
        for f_pump in channel_list:
            # 枚举f_pump，注意当信道功率为0时代表该信道没有信号
            index_pump = channel_list.index(f_pump)
            p_pump = channel_power[index_pump]
            if p_pump == 0:
                continue
            for freq in channel_list:
                # 枚举freq
                index_freq = channel_list.index(freq)
                # 不考虑泵浦光源对自身所在信道的影响
                if f_pump == freq:
                    continue
                if channel_dir[index_pump] == 'f':
                    p_sprs = self.raman_scatter_forward(p_pump, f_pump, freq)
                else:
                    p_sprs = self.raman_scatter_backward(p_pump, f_pump, freq)
                noise_power[index_freq] += p_sprs
        return noise_power

    def power_to_photon(self, p_noise: float, freq: float):
        """
        计算噪声所在信道freq上，非线性噪声对应的噪声光子数
        :param p_noise: 噪声功率（单位：W）
        :param freq: 噪声所在信道的信道中心频率（单位：Hz）
        :return: 计算得到的噪声光子数（单位：个）
        """
        return p_noise * self.detector_eff * self.gate_time * 10 ** (-0.1 * self.IL) / (PLANCK_CONSTANT * freq)


if __name__ == '__main__':
    a = Fiber(fiber_type='SMF')
    channelList = [193.4e12, 193.45e12, 193.5e12, 193.55e12, 193.6e12]
    channelPower = np.array([1e-3, 1e-3, 1e-3, 0, 0])
    channelDir = np.array(['b', 'b', 'b', 'b', 'b'])
    ap = a.calculate_sprs_noise(channelList, channelDir, channelPower)
    print(10 * np.log10(ap/1e-3))


    # # 寻找最小拉曼散射噪声信道
    # a = Fiber(fiber_type='HCF')
    # spacing = 0.1 * 1e12
    # fq = 193.4e12
    # cp_list = []
    # index_list = []
    # for i in range(-68, 56, 1):
    #     if i in [-7, -5, -3, -1, 1, 3, 5, 7]:
    #         continue
    #     channelList = [fq-3.5*spacing, fq-2.5*spacing, fq-1.5*spacing, fq-0.5*spacing, fq+i*spacing/2, fq+0.5*spacing, fq+1.5*spacing, fq+2.5*spacing, fq+3.5*spacing]
    #     index_list.append(i/2)
    #     channelPower = np.array([1e-3, 1e-3, 1e-3, 1e-3, 0, 1e-3, 1e-3, 1e-3, 1e-3])
    #     channelDir = np.array(['f', 'f', 'f', 'f', 'f', 'b', 'b', 'b', 'b'])
    #
    #     c = Fiber(fiber_type='HCF')
    #     cp = c.calculate_sprs_noise(channelList, channelDir, channelPower)[4]
    #     bp = c.calculate_fwm_noise(channelList, channelDir, channelPower)[4]
    #     cp_list.append(cp+bp)
    # max_value = max(cp_list)
    # max_index = cp_list.index(max_value)
    # print(index_list[max_index], max_value)
    # min_value = min(cp_list)
    # min_index = cp_list.index(min_value)
    # print(index_list[min_index], min_value)

    # cp = c.calculate_sprs_noise(channelList, channelDir, channelPower)[3]
    # print(cp)

    work_wave = 1550 * 1e-9
    gate_time = 1 * 10 ** (-9)
    spd_eff = 0.1
    IL = 8
    c_v = 299792458
    Planck_constant = 6.62607015 * 10 ** (-34)
    print(work_wave / c_v)
    print(1 / 193.4e12)
    # noise_num = cp * work_wave * gate_time * spd_eff * 10 ** (-0.1 * IL) / (Planck_constant * c_v)
    # print(noise_num)
    # # print(10 * np.log10(cp/1e-3))
