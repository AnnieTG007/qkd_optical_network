import numpy as np
import math

from numpy import ndarray
# k is the Boltzmann constant, and h the Planck constant
from scipy.constants import k, h, c

GNPY_RAMAN_COEFFICIENT = {
    # SSMF Raman coefficient profile in terms of mode intensity (g0 * A_ff_overlap)
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
    # SSMF Raman coefficient profile
    'g0': np.array(
        [0.00000000e+00, 1.12351610e-05, 3.47838074e-05, 5.79356636e-05, 8.06921680e-05, 9.79845709e-05, 1.10454361e-04,
         1.18735302e-04, 1.24736889e-04, 1.30110053e-04, 1.41001273e-04, 1.46383247e-04, 1.57011792e-04, 1.70765865e-04,
         1.88408911e-04, 2.05914127e-04, 2.24074028e-04, 2.47508283e-04, 2.77729174e-04, 3.08044243e-04, 3.34764439e-04,
         3.56481704e-04, 3.77127256e-04, 3.96269124e-04, 4.10955175e-04, 4.18718761e-04, 4.19511263e-04, 4.17025384e-04,
         4.13565369e-04, 4.07726048e-04, 3.83671291e-04, 4.08564283e-04, 3.69571936e-04, 3.14442090e-04, 2.16074535e-04,
         1.23097823e-04, 8.95457457e-05, 7.52470400e-05, 7.19806145e-05, 8.87961158e-05, 9.30812065e-05, 9.37058268e-05,
         8.45719619e-05, 6.90585286e-05, 4.50407159e-05, 3.36521245e-05, 3.02292475e-05, 2.69376939e-05, 2.60020897e-05,
         2.82958958e-05, 3.08667558e-05, 3.66024657e-05, 5.80610307e-05, 6.54797937e-05, 6.25022715e-05, 5.37806442e-05,
         3.94996621e-05, 2.68120644e-05, 2.33038554e-05, 1.79140757e-05, 1.52472424e-05, 1.32707565e-05, 1.06541760e-05,
         9.84649374e-06, 9.13999627e-06, 9.08971012e-06, 1.04227525e-05, 1.50419271e-05, 1.77838232e-05, 2.15810815e-05,
         2.03744008e-05, 1.81939341e-05, 1.31862121e-05, 9.65352116e-06, 8.62698322e-06, 9.18688016e-06, 1.01737784e-05,
         1.08017817e-05, 1.03903588e-05, 9.30040333e-06, 8.30809173e-06, 6.90650401e-06, 5.52238029e-06, 3.90648708e-06,
         2.22908227e-06, 1.55796177e-06, 9.77218716e-07, 3.23477236e-07, 1.60602454e-07, 7.97306386e-08]
    ),  # [1 / (W m)]

    # Note the non-uniform spacing of this range; this is required for properly capturing the Raman peak shape.
    'frequency_offset': np.array([
        0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5,
        12., 12.5, 12.75, 13., 13.25, 13.5, 14., 14.5, 14.75, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.25, 18.5, 18.75,
        19., 19.5, 20., 20.5, 21., 21.5, 22., 22.5, 23., 23.5, 24., 24.5, 25., 25.5, 26., 26.5, 27., 27.5, 28., 28.5,
        29., 29.5, 30., 30.5, 31., 31.5, 32., 32.5, 33., 33.5, 34., 34.5, 35., 35.5, 36., 36.5, 37., 37.5, 38., 38.5,
        39., 39.5, 40., 40.5, 41., 41.5, 42.]) * 1e12,  # [Hz]

    # Raman profile reference frequency
    'reference_frequency': 206.184634112792e12,  # [Hz] (1454 nm)

    # Raman profile reference effective area
    'reference_effective_area': 75.74659443542413e-12  # [m^2] (@1454 nm)
}

class Channel(object):
    def __init__(self, f_min: float = 191.3e12, f_max: float = 195.1e12, spacing: float = 0.25e9):
        # 为了简化问题，我们强制整个信道按照一个最小间隔进行等间隔采样，如果实际采样达不到就用插值填上
        # 频率最小值（单位：Hz）
        self.f_min = f_min
        # 频率最大值（单位：Hz）
        self.f_max = f_max
        # 信道频率间隔（单位：Hz）
        self.spacing = spacing
        # 信道频率列表
        self.channel_list = np.arange(self.f_min, self.f_max, self.spacing)
        # 信道数
        self.channel_num = len(self.channel_list)
        # 默认小编号节点发、大编号节点收为光信号传播+z方向
        # 信道状态 0禁用，1可用空闲，2经典信道，3量子信道
        self.channel_type = np.ones((self.channel_num, 2), dtype=np.int8)
        # 经典信号PSD建模(第一维度代表频率，第二维度代表方向，0正向，1反向，默认小号发大号收为正向)
        self.power_spectral_density = np.zeros((self.channel_num, 2), np.float64)

    @property
    def get_channel_list(self):
        return self.channel_list

    def get_power_spectral_density(self, direction: str):
        if direction == 'forward':
            return self.power_spectral_density[:,0]
        elif direction == 'backward':
            return self.power_spectral_density[:,1]
        elif direction == 'two-way':
            return self.power_spectral_density.sum(axis=1)
        else:
            NotImplementedError('direction should be "forward" , "backward" or "two-way", but read: ' + str(direction))
            return None

    def spacing_to_wave(self):
        f_aver = (self.f_max + self.f_min) / 2.0
        return c * self.spacing / f_aver ** 2


class Fiber(object):
    def __init__(self, channel: Channel = Channel()):
        # 光纤类型
        self._fiber_type = 'SMF'
        # 光纤长度
        self._len = 10e3
        # 光纤衰减(单位：m^-1)
        self._loss = 0.16 / 4.343 * 1e-3 * np.ones(channel.channel_num)
        self._channel = channel
        # 非线性系数(单位：W^-1*m^-1)
        self._non_linearity_coefficient = 5.0e-7
        # 信道频率
        self._channel = channel
        # 色散系数(单位：s/m^2，还是s^2/m？)
        self._dispersion_coefficient = 2.0e-6
        # 色散斜率(单位：s/m^3，还是s^2/m？)
        self._dispersion_slope = 4e-3
        # 工作温度（单位：K）
        self._temperature = 300
        # 模场直径（单位：m）
        self._mode_field_diameter = 10.5e-6
        # 有效模面积（单位：m^2）
        self._effective_area = self.get_effective_area
        self.n = 1.45

    @property
    def get_dispersion_coefficient(self):
        return self._dispersion_coefficient

    @property
    def get_dispersion_slope(self):
        return self._dispersion_slope

    @property
    def get_effective_area(self):
        return math.pi * self._mode_field_diameter ** 2 / 4.0

    @property
    def get_temperature(self):
        return self._temperature

    @property
    def get_channel(self):
        return self._channel

    @property
    def get_loss(self):
        return self._loss


class NoiseSolver(object):
    def __init__(self, fiber: Fiber = Fiber()):
        self._fiber = fiber

class RamanSolver(NoiseSolver):
    def __init__(self, raman=None, fiber: Fiber = Fiber()):
        super().__init__(fiber)
        if raman is None:
            self._raman = GNPY_RAMAN_COEFFICIENT
        else:
            self._raman = raman

    def _get_co_raman_scatter(self, pump_channel: np.ndarray, signal_channel: np.ndarray):
        """
        求解受激拉曼噪声和自发拉曼散射噪声共需的系数
        :param pump_channel: np.ndarray, pump信道频率 shape = (1， Nc)
        :param signal_channel: np.ndarray, signal信道频率, shape = (Nq, 1)
        :return:
        """
        # 将signal_channel整理为大小为(Nq,1)的列向量
        freq_q = signal_channel.reshape(-1, 1)
        # 将pump_channel整理为大小为(1, Nc)的行向量
        freq_c = pump_channel.reshape(1, -1)
        # 利用numpy的广播功能计算大小为(Nq,Nc)的频率差矩阵
        # 在QKD共纤传输系统中，注意泵浦光对应经典信号，信号光对应量子信号
        df = freq_q - freq_c
        # freq_q和freq_c的比值，便于anti_stoke过程的系数补偿
        div_f = freq_q / freq_c

        # 斯托克斯效应掩码，散射光频率更低
        df_stoke_mask = np.where(df < 0, 1, 0.0)
        # 反斯托克斯效应掩码，散射光频率更高
        df_anti_stoke_mask = np.where(df > 0, 1, 0.0)

        # 矩阵运算时，freq_q会自动沿着列方向管广播，由(Nq, 1)广播为(Nq, Nc)
        gamma_raman = self._get_raman_gain_coefficient(freq_c, freq_q)
        # 对于freq_c == freq_q的情况，通过df_stoke_mask和df_anti_stoke_mask掩码约束，将有效拉曼截面记为0
        stoke_coff = df_stoke_mask * gamma_raman * self._fiber.get_channel.spacing
        # 计算反stoke量的时候，需要补偿频率比值
        # 对比reference.md [1] 的Eq.(5)的anti_stoke补偿系数，默认群速度不随频率变化，也就是折射率不随频率变化
        # 如果后续考虑折射率随频率的变化，需要修改这里的表述
        anti_stoke_coff = df_anti_stoke_mask * gamma_raman * div_f * self._fiber.get_channel.spacing

        return freq_q, df, stoke_coff, anti_stoke_coff

    def _get_raman_gain_coefficient(self, freq_c: ndarray, freq_q: ndarray):
        """
        根据频移 |freq_q - freq_c| 对 拉曼增益系数 插值
        :param freq_c: np.ndarray, 经典信道列表, shape = (1, Nc)
        :param freq_q: np.ndarray, 查询信道列表（可以是经典/量子信道）, shape = (Nq, 1)
        :return:
        """
        delta_f = np.abs(freq_q - freq_c)

        gamma = np.interp(
            delta_f,
            self._raman['frequency_offset'],
            self._raman['g0'],
            left=0.0,
            right=0.0
        )
        # 依照泵浦光频率进行修正
        edit_gamma = (gamma * freq_c / self._raman['reference_frequency']
                      * self._raman['reference_effective_area'] / self._fiber.get_effective_area)
        return edit_gamma

    def _get_sprs_coff(self, pump_channel, signal_channel: np.ndarray):
        """
        计算自发拉曼散射系数，单位m^(-1)
        :param pump_channel: np.ndarray, pump信道频率 shape = (1， Nc)
        :param signal_channel: np.ndarray, signal信道频率, shape = (Nq, 1)
        :return: sprs_coff : np.ndarray, shape = (Nq, Nc)
        每个元素对应 经典采样频率 fc_j -> 查询频率 fq_i 的自发拉曼散射系数
        """
        # 首先计算和受激拉曼散射的共用系数
        freq_q, df, stoke_coff, anti_stoke_coff = self._get_co_raman_scatter(pump_channel, signal_channel)
        # 对df的每个元素取绝对值
        delta_df = np.abs(df)

        # 计算有效拉曼截面，对应reference.md [1] 的Eq.(5)，用于计算reference.md [2] 的 Eq.(A1)
        # 只计算df != 0(即绝对值delta_df>0)时的eta值，避免运算时出现nan
        eta = np.zeros_like(delta_df, dtype=np.float64)
        nonzero_mask = delta_df > 0
        eta[nonzero_mask] = 1.0 / (np.exp(h * delta_df[nonzero_mask] / (k * self._fiber.get_temperature)) - 1)

        # 对于freq_c == freq_q的情况，通过df_stoke_mask和df_anti_stoke_mask掩码约束，将有效拉曼截面记为0
        stoke_section = 2 * h * freq_q * stoke_coff * (1 + eta)
        # 计算反stoke量的时候，需要补偿频率比值
        # 对比reference.md [1] 的Eq.(5)的anti_stoke补偿系数，默认群速度不随频率变化，也就是折射率不随频率变化
        # 如果后续考虑折射率随频率的变化，需要修改这里的表述
        anti_stoke_section = 2 * h * freq_q * anti_stoke_coff * eta

        return stoke_section + anti_stoke_section

    def _get_srs_coff(self, pump_channel: np.ndarray, signal_channel: np.ndarray):
        """
        计算受激拉曼散射系数，单位m^(-1)
        :param pump_channel: np.ndarray, pump信道频率 shape = (1， Nc)
        :param signal_channel: np.ndarray, signal信道频率, shape = (Nq, 1)
        每个元素对应 经典采样频率 fc_j -> 查询频率 fq_i 的自发拉曼散射系数
        """
        # 首先计算和自发拉曼散射的共用系数
        freq_q, df, stoke_coff, anti_stoke_coff = self._get_co_raman_scatter(pump_channel, signal_channel)

        # 对于freq_c == freq_q的情况，通过df_stoke_mask和df_anti_stoke_mask掩码约束，将有效拉曼截面记为0
        stoke_section = stoke_coff
        anti_stoke_section = anti_stoke_coff
        return stoke_section - anti_stoke_section

    def get_intra_core_raman_noise(self, pump_channel: np.ndarray, signal_channel: np.ndarray, z: np.ndarray, direction: str= 'forward'):
        """
        计算单芯拉曼散射噪声功率
        :param pump_channel: np.ndarray, pump信道频率 shape = (1， Nc)
        :param signal_channel: np.ndarray, signal信道频率, shape = (Nq, 1)
        :param z: np.ndarray，光纤长度, shape = (1, Nz)
        :param direction: str，定义噪声类别，'forward' 代表前向自发拉曼散射噪声，'backward' 代表后向自发拉曼散射噪声
        :return: np.ndarray，前向拉曼散射噪声功率，shape = (Nq, 1)， 单位为 W
        """
        def forward_raman(g, signal_loss, pump_loss, loss_zero_mask, fiber_len):
            """
            计算前向拉曼散射噪声功率，即对之前计算的前向拉曼散射噪声对距离的微分结果，再沿着+z方向积分即可
            参考文献 reference.md [3]中的Eq.(9)
            :param g: np.ndarray, 前向拉曼散射噪声对距离的微分 (Nq, Nc, 1)
            :param signal_loss: np.ndarray, signal光在光纤中的损耗，考虑损耗波长依赖性 shape = (Nq, 1. 1)
            :param pump_loss: np.ndarray, pump光在光纤中的损耗，考虑损耗波长依赖性 shape = (1, Nc. 1)
            :param loss_zero_mask: np.ndarray, signal光和pump光损耗相同的掩码 shape = (Nq, Nc, 1)
            :param fiber_len: np.ndarray，光纤长度, shape = (1, 1, Nz)
            :return: 前向拉曼散射噪声功率随距离的变化(Nq, z)
            """
            # 定义noise_matrix矩阵，便于运算 shape = (Nq, Nc, Nz)
            noise_matrix = np.zeros((loss_zero_mask.shape[0], loss_zero_mask.shape[1], fiber_len.shape[2]), dtype=np.float64)
            # 如果pump光和signal光的损耗相同
            noise_matrix[loss_zero_mask] = (g * np.exp(-signal_loss * fiber_len) * fiber_len)[loss_zero_mask]
            # 如果pump光和signal光的损耗不同，1e9无含义只是为了占位避免除0警告
            noise_matrix[~loss_zero_mask] = (g * (np.exp(-pump_loss * fiber_len) - np.exp(-signal_loss * fiber_len))
                                             / np.where(signal_loss == pump_loss, 1e9, signal_loss - pump_loss))[~loss_zero_mask]
            # 返回结果时应对Nc维度求和，考虑所有泵浦光的贡献
            return np.sum(noise_matrix, axis=1)

        def backward_raman(g, signal_loss, pump_loss, loss_zero_mask, fiber_len):
            """
            计算后向拉曼散射噪声功率，即对之前计算的后向拉曼散射噪声对距离的微分结果，再沿着-z方向积分即可
            参考文献 reference.md []中的Eq.()
            :param g: np.ndarray, 前向拉曼散射噪声对距离的微分 (Nq, Nc, 1)
            :param signal_loss: np.ndarray, signal光在光纤中的损耗，考虑损耗波长依赖性 shape = (Nq, 1. 1)
            :param pump_loss: np.ndarray, pump光在光纤中的损耗，考虑损耗波长依赖性 shape = (1, Nc. 1)
            :param loss_zero_mask: np.ndarray, signal光和pump光损耗相同的掩码 shape = (Nq, Nc, 1)
            :param fiber_len: np.ndarray，光纤长度, shape = (1, 1, Nz)
            :return: 后向拉曼散射噪声功率随距离的变化(Nq, z)
            """
            # 定义noise_matrix矩阵，便于运算 shape = (Nq, Nc, Nz)
            noise_matrix = np.zeros((loss_zero_mask.shape[0], loss_zero_mask.shape[1], fiber_len.shape[2]),
                                    dtype=np.float64)
            # 如果pump光和signal光的损耗相同
            noise_matrix[loss_zero_mask] = (g * (1 - np.exp(-2.0 * signal_loss * fiber_len)) / (2.0 * signal_loss))[loss_zero_mask]
            # 如果pump光和signal光的损耗不同
            noise_matrix[~loss_zero_mask] = (g * (1 - np.exp(-(signal_loss + pump_loss) * fiber_len)) / (signal_loss + pump_loss))[~loss_zero_mask]
            # 返回结果时应对Nc维度求和，考虑所有泵浦光的贡献
            return np.sum(noise_matrix, axis=1)

        # -----------------考虑波长依赖性的损耗计算-----------------
        # 查找pump_channel中每个元素在channel_list中的下标，其中numpy.ravel()用于展平矩阵
        idx_pump = np.searchsorted(self._fiber.get_channel.channel_list.ravel(), pump_channel.ravel())
        # 查找signal_channel中每个元素在channel_list中的下标，其中numpy.ravel()用于展平矩阵
        idx_signal = np.searchsorted(self._fiber.get_channel.channel_list.ravel(), signal_channel.ravel())
        # 考虑损耗的波长依赖性，signal信号的损耗矩阵 shape = (Nq, 1)
        loss_signal = self._fiber.get_loss[idx_signal].reshape(-1, 1)
        # 考虑损耗的波长依赖性，pump信号的损耗矩阵 shape = (1, Nc)
        loss_pump = self._fiber.get_loss[idx_pump].reshape(1, -1)
        # signal信号和pump信号的损耗差矩阵 shape = (Nq, Nc)
        delta_loss = loss_signal - loss_pump
        # 判断signal信号和pump信号的损耗是否相同
        delta_loss_zero_mask = abs(delta_loss) < 1e-9

        # -----------------拉曼散射噪声功率矩阵对距离的微分计算-----------------
        # 先求解自发拉曼散射噪声对距离的微分， shape = (Nq, Nc)
        sprs_coff = self._get_sprs_coff(pump_channel, signal_channel)
        # 然后求解受激拉曼散射噪声对距离的微分， shape = (Nq, Nc)
        srs_coff = self._get_srs_coff(pump_channel, signal_channel)
        if direction == 'forward':
            func = forward_raman
        elif direction == 'backward':
            func = backward_raman
        else:
            raise NotImplementedError('direction should be "forward" or "backward", but read: ' + str(direction))

        # 提取采样频率点的pump信号功率密度正反向求和，整理为shape = (1, Nc)的形式
        power_pump = self._fiber.get_channel.get_power_spectral_density(direction='two-way')[idx_pump].reshape(1, -1)
        # 提取采样频率点的signal信号功率密度正向/反向，整理为shape = (Nq, 1)的形式
        power_signal = self._fiber.get_channel.get_power_spectral_density(direction=direction)[idx_signal].reshape(-1, 1)

        # 求解自发拉曼散射噪声功率矩阵对距离的微分，并对距离积分， shape = (Nq, Nc)
        sprs_noise = func((sprs_coff * power_pump)[:, :, None] * self._fiber.get_channel.spacing,
                          loss_signal.reshape(-1, 1, 1), loss_pump.reshape(1, -1, 1), delta_loss_zero_mask[:, :, None], z[None, None, :])
        # 求解受激拉曼散射噪声功率矩阵对距离的微分，并对距离积分， shape = (Nq, Nc)
        srs_noise = func((srs_coff * power_signal * power_pump)[:, :, None] * self._fiber.get_channel.spacing,
                          loss_signal.reshape(-1, 1, 1), loss_pump.reshape(1, -1, 1), delta_loss_zero_mask[:, :, None], z[None, None, :])

        return sprs_noise


class FWMSolver(NoiseSolver):
    def __init__(self, fiber: Fiber = Fiber()):
        super().__init__(fiber)

    def _get_fwm_combination(self, pump_channel: np.ndarray, signal_channel: np.ndarray, direction: str = 'forward'):
        """
        给出能产生signal信道频率的pump光组合，形式为channel_i, channel_j, channel_k, channel_fwm
        满足 channel_i + channel_j - channel_k == channel_fwm
        :param pump_channel: np.ndarray, pump信道频率 shape = (1， Nc)
        :param signal_channel: np.ndarray, signal信道频率, shape = (Nq, 1)
        :param direction:
        :return: np.ndarray，
        """
        # 查找pump_channel中每个元素在channel_list中的下标，其中numpy.ravel()用于展平矩阵
        idx_pump = np.searchsorted(self._fiber.get_channel.channel_list.ravel(), pump_channel.ravel())
        # 查找signal_channel中每个元素在channel_list中的下标，其中numpy.ravel()用于展平矩阵
        idx_signal = np.searchsorted(self._fiber.get_channel.channel_list.ravel(), signal_channel.ravel())

        # 这里取到所有pump_channel两两组合的情况
        num_pump = idx_pump.size
        num_signal = idx_signal.size
        idx_pump_i = np.repeat(np.arange(num_pump), num_pump)  # pump_channel 内局部下标
        idx_pump_j = np.tile(np.arange(num_pump), num_pump)  # pump_channel 内局部下标

        # 对pump光频率两两求和，因为我们这里是等间隔采样，所以有f_{i+j}=f_{i} + f_{j}，用编号求和代替频率求和
        # 注意，通过idx_pump[idx_pump_i]将pump_channel的局部下标映射到channel中的全局下标
        pair_sum = idx_pump[idx_pump_i] + idx_pump[idx_pump_j]

        # 在pump光中查找是否有满足pump_k == pump_i + pump_j - signal_fwm的对象
        # 用target表示signal_fwm + pump_k，然后去和pump_i+pump_j列表里进行匹配
        idx_signal_fwm = np.repeat(np.arange(num_signal), num_pump) # signal_channel 内局部下标
        idx_pump_k = np.tile(np.arange(num_pump), num_signal) # pump_channel 内局部下标
        # 构造匹配目标 pump_k + signal_fwm，映射到channel中的全局下标
        target = idx_signal[idx_signal_fwm] + idx_pump[idx_pump_k]

        # 对pump光频率两两求和结果排序并匹配目标结果
        order = np.argsort(pair_sum)
        pair_sum_sorted = pair_sum[order]
        left = np.searchsorted(pair_sum_sorted, target, side='left')
        right = np.searchsorted(pair_sum_sorted, target, side='right')
        counts = right - left
        print(counts)

        # 筛选产生有效fwm分量的signal信道
        valid_mask = counts > 0
        counts_valid = counts[valid_mask]
        left_valid = left[valid_mask]

        # 展开匹配项
        total_matches = counts_valid.sum()
        starts = np.cumsum(counts_valid) - counts_valid
        offsets = np.arange(total_matches) - np.repeat(starts, counts_valid)
        pos_in_sorted = np.repeat(left_valid, counts_valid) + offsets
        matched_pair_idx = order[pos_in_sorted]
        matched_target_idx = np.repeat(np.flatnonzero(valid_mask), counts_valid)

        # 还原组合下标，先通过idx_pump_i映射到pump_channel的编号中，再通过idx_pump映射到channel全局编号中
        idx_channel_i = idx_pump[idx_pump_i[matched_pair_idx]]
        idx_channel_j = idx_pump[idx_pump_j[matched_pair_idx]]
        idx_channel_k = idx_pump[idx_pump_k[matched_target_idx]]
        idx_signal_fwm = idx_signal_fwm[matched_target_idx]
        idx_channel_fwm = idx_signal[idx_signal_fwm]

        return idx_channel_i, idx_channel_j, idx_channel_k, idx_channel_fwm, idx_signal_fwm

    def get_intra_core_fwm_noise(self, pump_channel: np.ndarray, signal_channel: np.ndarray, fiber_len: np.ndarray, direction: str = 'forward'):
        """
        计算单芯拉曼散射噪声功率
        :param pump_channel: np.ndarray, pump信道频率 shape = (1， Nc)
        :param signal_channel: np.ndarray, signal信道频率, shape = (Nq, 1)
        :param fiber_len: np.ndarray，光纤长度, shape = (1, Nz)
        :param direction: str，定义噪声类别，'forward' 代表前向自发拉曼散射噪声，'backward' 代表后向自发拉曼散射噪声
        :return: np.ndarray，前向拉曼散射噪声功率，shape = (Nq, 1)， 单位为 W
        """
        # 获取产生有效四波混频分量的频率组合（返回的是信道在channel中的编号）
        idx_channel_i, idx_channel_j, idx_channel_k, idx_channel_fwm, idx_signal_fwm = self._get_fwm_combination(
            pump_channel, signal_channel)

        # ------------计算direction向pump光直接引起的前向四波混频噪声------------
        # 读取pump光的功率，由reference [4] 可知，与拉曼散射噪声不同，FWM效应要求相位匹配，包括pump光相同的传播方向
        # power_pump_i, power_pump_i, power_pump_i, shape = (Nf, 1)
        power_pump_all = self._fiber.get_channel.get_power_spectral_density(direction=direction)
        power_pump_i = power_pump_all[idx_channel_i]
        power_pump_j = power_pump_all[idx_channel_j]
        power_pump_k = power_pump_all[idx_channel_k]

        # 计算系数（暂时不知道是什么系数）
        coff_fwm = self._get_fwm_coff(idx_channel_fwm)

        # 计算四波混频效率
        fwm_eff = self._get_fwm_efficiency_factor(idx_channel_i, idx_channel_j, idx_channel_k, idx_channel_fwm, fiber_len)

        # 所有组合对f_fwm的贡献
        f_fwm = coff_fwm * power_pump_i * power_pump_j * power_pump_k * fwm_eff

        # 按 idx_signal_fwm 聚合到 ans 的各行
        ans = np.zeros((signal_channel.size, fiber_len.size), dtype=np.float64)
        np.add.at(ans, idx_signal_fwm, f_fwm)

        # ------------计算与direction反向pump光引起的后向四波混频噪声------------
        if direction == 'forward':
            anti_direction = 'backward'
        elif direction == 'backward':
            anti_direction = 'forward'
        else:
            raise Exception('direction must be either "forward" or "backward"')

        anti_power_pump_all = self._fiber.get_channel.get_power_spectral_density(direction=anti_direction)
        anti_power_pump_i = power_pump_all[idx_channel_i]
        anti_power_pump_j = power_pump_all[idx_channel_j]
        anti_power_pump_k = power_pump_all[idx_channel_k]

        return 0

    @staticmethod
    def _get_delta_alpha(loss_i: np.ndarray, loss_j: np.ndarray, loss_k: np.ndarray, loss_fwm: np.ndarray):
        """
        :param loss_i: np.ndarray， pump_i的衰减，shape = (Nf, 1)
        :param loss_j: np.ndarray， pump_j的衰减，shape = (Nf, 1)
        :param loss_k: np.ndarray， pump_k的衰减，shape = (Nf, 1)
        :param loss_fwm: np.ndarray， signal_fwm的衰减，shape = (Nf, 1)
        :return: delta_alpha, np.ndarray, shape = (Nf, 1)
        """
        return loss_i + loss_j + loss_k - loss_fwm

    def _get_phase_matching_factor(self, channel_i: np.ndarray, channel_j: np.ndarray, channel_fwm: np.ndarray):
        """
        求解相位匹配因子，注意这里的输入要求是真实频率值而不是信道编号
        其中Nf表示组合成signal_channel的组合数
        :param channel_i: np.ndarray， pump_i，shape = (Nf, 1)
        :param channel_j: np.ndarray， pump_j, shape = (Nf, 1)
        :param channel_fwm: np.ndarray， f = fi + fj - fk， shape = (Nf, 1)
        :return: 相位匹配因子， np.ndarray, shape = (Nf, 1)
        """
        # 求解channel_k， shape = (Nf, 1)
        # 注意经过前面的组合预处理之后，channel_i，channel_j，channel_k还有channel_fwm尺寸相同
        channel_k = channel_i + channel_j - channel_fwm
        lamda_k = c / channel_k
        delta_ik = np.abs(channel_i - channel_k)
        delta_jk = np.abs(channel_j - channel_k)
        return (2 * np.pi * lamda_k ** 2 / c * delta_ik * delta_jk *
                (self._fiber.get_dispersion_coefficient + lamda_k ** 2 / 2 / c * (delta_ik + delta_jk) * self._fiber.get_dispersion_coefficient))

    def _get_fwm_efficiency_factor(self, idx_channel_i, idx_channel_j, idx_channel_k, idx_channel_fwm, fiber_len):
        """
        计算四波混频效率
        :param idx_channel_i: np.ndarray， pump_i，shape = (Nf, 1)
        :param idx_channel_j: np.ndarray， pump_j, shape = (Nf, 1)
        :param idx_channel_k: np.ndarray， pump_k, shape = (Nf, 1)
        :param idx_channel_fwm: np.ndarray， f = fi + fj - fk， shape = (Nf, 1)
        :param fiber_len: np.ndarray，光纤长度, shape = (1, Nz)
        :return: fwm_efficiency_factor， np.ndarray, shape = (Nf, Nz)
        """
        # 将信道在channel中的编号转换为频率列表
        channel_i = self._fiber.get_channel.channel_list[idx_channel_i]
        channel_j = self._fiber.get_channel.channel_list[idx_channel_j]
        channel_fwm = self._fiber.get_channel.channel_list[idx_channel_fwm]

        # 考虑损耗的波长依赖性，获取pump光和signal光的损耗矩阵，shape = (Nf, 1)
        loss_i = self._fiber.get_loss[idx_channel_i].reshape(-1, 1)
        loss_j = self._fiber.get_loss[idx_channel_j].reshape(-1, 1)
        loss_k = self._fiber.get_loss[idx_channel_k].reshape(-1, 1)
        loss_fwm = self._fiber.get_loss[idx_channel_fwm].reshape(-1, 1)

        # 获取delta_alpha （暂时还不知道这个物理量应该如何命名），shape = (Nf, 1)
        delta_alpha = self._get_delta_alpha(loss_i=loss_i, loss_j=loss_j, loss_k=loss_k, loss_fwm=loss_fwm)

        # 获取相位匹配因子，shape = (Nf, 1)
        phase_matching_factor = self._get_phase_matching_factor(channel_i=channel_i, channel_j=channel_j, channel_fwm=channel_fwm)

        # 确保fiber_len.shape==(1, Nz)，才能触发python的广播机制而不是尺寸不匹配的报错
        fiber_len = fiber_len.reshape(1, -1)

        # 为了表达简洁，将四波混频效率拆分为分子和分母两部分
        # 分子部分
        rtn_numerator = np.exp(delta_alpha * fiber_len) + 1 - 2 * np.exp(delta_alpha * fiber_len / 2) * np.cos(phase_matching_factor * fiber_len)
        # 分母部分
        rtn_denominator = delta_alpha ** 2 / 4 + phase_matching_factor ** 2
        return rtn_numerator / rtn_denominator

    def _get_fwm_coff(self, channel_fwm: np.ndarray):
        return 256 * np.pi ** 4 / self._fiber.get_n ** 4 / c ** 4 / self._fiber.get_effective_area

if __name__ == '__main__':
    # channel configuration
    f_min_test = 180.0e12
    f_max_test = 200.0e12
    spacing_test = 20e9
    channel_obj = Channel(f_min=f_min_test, f_max=f_max_test, spacing=spacing_test)

    #classic_channel = np.array([193.4, 193.5, 193.6, 193.7]) * 1e12
    classic_channel = []
    P0 = 1e-3

    for i in range(channel_obj.channel_num):
        channel_temp = channel_obj.channel_list[i]
        if 191.1e12 < channel_temp < 191.2e12:
            classic_channel.append(channel_temp)

    classic_channel = np.array(classic_channel)

    for i in range(channel_obj.channel_num):
        channel_temp = channel_obj.channel_list[i]
        if channel_temp in classic_channel:
            channel_obj.channel_type[i][0] = 2
            channel_obj.power_spectral_density[i][0] = P0 / 0.1e12
        else:
            channel_obj.channel_type[i][0] = 3

    # fiber_configuration
    fiber_obj = Fiber(channel=channel_obj)
    fwm_solver = FWMSolver(fiber=fiber_obj).get_intra_core_fwm_noise(pump_channel=classic_channel, signal_channel=channel_obj.channel_list,
                                                                     fiber_len=np.array([10e3]), direction='forward')


    # raman_solver = RamanSolver(raman=None, fiber=fiber_obj)
    # noise = raman_solver.get_intra_core_raman_noise(pump_channel=classic_channel, signal_channel=channel_obj.channel_list,
    #                                                 z=np.array([10e3]), direction='forward')
    #
    # import matplotlib.pyplot as plt
    # x = channel_obj.channel_list
    # y = (noise.reshape(-1,1)) # + channel_obj.get_power_spectral_density(direction='forward')
    # plt.figure()
    # plt.plot(x, y)
    # plt.yscale('log')
    # plt.show()
