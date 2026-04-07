import pandas as pd
import numpy as np
import random
import math
from copy import deepcopy

import algorithm
import topology


def gen_wave(num_of_channels: int, interval: float):
    """
    :param num_of_channels: 网络中可用的波长数目
    :param interval: 信道间隔
    :return: 网络中的各个波长，从1550nm
    """
    if num_of_channels % 2 == 0 and num_of_channels != 0:    # 此时为偶数用下面的波长列表以1550nm为中心向两侧展开但是不包括1550nm
        wave_channels_forward = [193.5*1e12 - (math.floor(num_of_channels/2)-i) * interval for i in range(math.floor(num_of_channels/2))]
        wave_channels_backward = [193.5*1e12 + i * interval for i in range(math.floor(num_of_channels/2))]
        wave_channels1 = wave_channels_forward + wave_channels_backward
    elif num_of_channels % 2 == 1:
        wave_channels_forward = [193.5 * 1e12 - (math.floor(num_of_channels / 2) - i) * interval for i in range(math.floor(num_of_channels / 2))]
        wave_channels_mid = [193.5 * 1e12]
        wave_channels_backward = [193.5 * 1e12 + (i+1) * interval for i in range(math.floor(num_of_channels / 2))]
        wave_channels1 = wave_channels_forward + wave_channels_mid + wave_channels_backward
    else:
        raise ValueError('Please check whether the input is an integer(+)')   # 信道数必须为正整数

    if wave_channels1[-1] > 196.0*1e12 or wave_channels1[0] < 191.7*1e12:
        raise ValueError("The channel is out of C-band")
    else:
        return wave_channels1


class Event:
    """定义业务到达或离去事件"""
    def __init__(self):
        self.m_eventType = pd.Series([0, 0], index=['Arrival', 'End'])                         # 事件的类型(业务到达或离去)
        self.m_time = 0                                                                             # 事件的触发事件
        self.m_holdTime = 0                                                                         # 业务的持续时间
        self.m_id = 0                                                                               # 业务id
        self.m_sourceNode = 0                                                                       # 业务源节点
        self.m_destNode = 0                                                                         # 业务目的节点
        self.m_occupiedWave = 0                                                                     # 业务占有波长的编号
        self.m_workPath = []                                                                        # 业务的完整路由
        self.P = 1e-3                                                                               # 业务的功率，暂时为定值

    def generate_leaving_event(self):
        event1 = deepcopy(self)                                                                     # 非常重要，必须利用深拷贝
        event1.m_eventType["End"] = 1
        event1.m_eventType["Arrival"] = 0
        event1.m_time = self.m_time + self.m_holdTime                                               # "业务离去事件的发生时间" = "业务到达事件发生的时间" + "业务持续时间"
        return event1


class ClassicalService:
    """定义网络和业务数据"""
    def __init__(self, TOPOLOGY=topology.net1, WaveNumber=16, Ts=3, lambda1=1600, rou=0.5):
        self.network = TOPOLOGY[0]                                                                  # 存放网络拓扑（单位：m）
        self.fiber_type = TOPOLOGY[1]                                                               # 存放网络光纤分布
        self.MAXINUM = self.network.shape[0]                                                        # 网络拓扑节点个数
        self.WaveNumber = WaveNumber                                                                # 密集波分复用系统波长数
        self.wave_interval = 50e9                                                                   # 信道间隔（单位: Hz）
        self.available_channel = gen_wave(self.WaveNumber, self.wave_interval)                      # 可用波长列表

        # 光纤链路资源矩阵，维度代表链路起点、终点、波长。其中0为不可用，1为可用，2为正在被使用，3为量子信道。
        self.m_resourceMap = np.ones((self.MAXINUM, self.MAXINUM, self.WaveNumber), dtype=np.float32)
        # 每个经典波长上所需的安全密钥率
        self.m_skdMap = np.zeros((self.MAXINUM, self.MAXINUM, self.WaveNumber), dtype=np.float32)
        # 每个波长上的光功率
        self.P_link = np.zeros((self.MAXINUM, self.MAXINUM, self.WaveNumber), dtype=np.float32)
        # 各链路QKD收发机数量，每个链路不一样
        self.QKD_transceiver = np.zeros((self.MAXINUM, self.MAXINUM), dtype=int)

        self.Ts = Ts                                                                                # 时隙个数
        self.m_currentTime = 0                                                                      # 当前时间
        self.lambda1 = lambda1                                                                      # 业务的到达率（产生到达时间）
        self.m_lambda1 = 1.0/self.lambda1                                                           # 泊松分布转化为指数分布，到达率要取反
        self.m_rou1 = rou                                                                           # 业务的离去率（产生离去时间），这个数的意义就是业务持续时间的期望
        self.erlang = self.lambda1 * self.m_rou1                                                    # 业务量计算

        self.ServiceQuantity = 0                                                                    # 业务总数
        self.num_arrival = 0                                                                        # 到达事件数量
        self.num_leaving = 0                                                                        # 离去事件数量
        self.m_sunOfFailedService = 0                                                               # 被阻塞的业务数量
        self.m_nextServiceId = 0                                                                    # 新一次服务的业务id
        self.m_pq = []                                                                              # 事件的容器
        self.m_sumOfFailedService = 0

        self.pa_unseen = 0
        self.seed = 50
        if self.pa_unseen:
            np.random.seed(314159)
            random.seed(self.seed)
        else:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.c_s_amount = []                                                            # 各个时刻网络总体的业务数量
        self.s_timeList = []                                                            # 各个时刻网络各链路业务量的和，与c_s_amount是不同的
        self.SKD_timeList = []                                                          # 各个时刻网络总体的SKD
        self.SKR_timeList = []                                                          # 各个时刻网络总体的SKR
        self.Link_SKR_timeList = np.zeros((self.Ts, self.MAXINUM, self.MAXINUM))        # 这个数组存储每个时刻的网络各个链路的SKR
        self.Link_s_timeList = []                                                       # 这个数组存储每个时刻的网络各个链路的业务数量
        self.SKD = [1e3, 2e3, 4e3]
        self.SKD_weights = [0.25, 0.25, 0.5]
        self.Link_SKD_timeList = np.zeros((self.Ts, self.MAXINUM, self.MAXINUM))        # 这个数组存储每个时刻的网络各个链路的SKD
        self.ClassicNoise = np.zeros(self.Ts)

        self.network_init()  # 初始化网络参数

    def network_init(self):
        """
        网络初始化，设置量子信道
        :return: none
        """
        for i in range(self.MAXINUM):
            for j in range(self.MAXINUM):
                if i == j:
                    continue
                self.m_resourceMap[i][j][0] = 3
        print('Initialization is done')
        return

    def generateServiceEventPair(self, event_id):
        """
        生成业务，更新业务列表m_pq
        :param event_id: event_id
        :return: None,but update the list named m_pq
        """
        event0 = Event()
        # 业务到达
        event0.m_eventType["Arrival"] = 1
        event0.m_eventType["End"] = 0
        event0.m_id = event_id
        event0.m_time = self.m_currentTime + self.arrive_time_gen()
        # 暂时定义功率固定
        event0.P = 1e-3
        # 业务持续时间
        event0.m_holdTime = self.arrive_time_gen()

        # 为业务分配SKD，调整权重是关键
        event0.secure_level = random.choices(self.SKD, weights=self.SKD_weights, k=1)

        # 为event0事件生成源目的节点
        node = self.randomSrcDst()
        event0.m_sourceNode = node["first"]
        event0.m_destNode = node["second"]

        # 将event0加入到m_pq中
        self.m_pq.append(event0)
        # 排序会让先到达的排在前面，会耗费时间
        self.m_pq.sort(key=lambda x: x.m_time)

    def randomSrcDst(self):
        """
        随机生成源节点和目的节点，且源节点和目的节点不能相同
        :return: 源节点目的节点
        """
        SrcDst = pd.Series([0, 0], index=['first', 'second'])
        SrcDst['second'] = random.randint(0, self.MAXINUM - 1)
        while True:
            SrcDst['first'] = random.randint(0, self.MAXINUM - 1)
            if SrcDst['first'] != SrcDst['second']:
                break
        return SrcDst

    def arrive_time_gen(self):
        """
        生成业务到达离去时间，都是生成一个服从指数分布的时间差
        :return: 到达离去时间
        """
        while True:
            u = random.random()   # 0~1的范围的整数
            if u != 0:
                break
        ln = math.log(u)          # 默认底数为e
        x = self.m_lambda1*ln
        x *= (-1)
        return x

    def dealWithEvent(self, event):
        """
        处理时间的到达、离去事件
        :return: none
        """
        # 当前时间=业务的到达时间/离去时间
        self.m_currentTime = event.m_time
        if event.m_eventType['Arrival'] == 1:
            self.ServiceQuantity += 1
            self.num_arrival += 1

            reg_wave, reg_path = self.resource_allocation(event)

            if reg_wave == -1:
                # 业务阻塞，统计阻塞率
                self.m_sumOfFailedService += 1
            else:
                # 为业务分配资源
                event.m_occupiedWave = reg_wave
                event.m_workPath = reg_path
                # 标记链路属性为正在被占用
                for it in range(len(event.m_workPath) - 1):
                    self.m_resourceMap[event.m_workPath[it]][event.m_workPath[it + 1]][event.m_occupiedWave] = 2
                    self.P_link[event.m_workPath[it]][event.m_workPath[it + 1]][event.m_occupiedWave] = event.P
                    self.m_skdMap[event.m_workPath[it]][event.m_workPath[it + 1]][event.m_occupiedWave] = event.secure_level[0]

                # 生成该业务的离去事件
                self.m_pq.append(event.generate_leaving_event())
                # 优化排序，特别耗费时间
                self.m_pq.sort(key=lambda x: x.m_time)

            # 生成下一个到达事件
            self.generateServiceEventPair(self.m_nextServiceId)
            self.m_nextServiceId += 1
            # 删除当前事件
            self.m_pq.remove(event)

        # 业务离去事件
        if event.m_eventType['End'] == 1:
            self.num_leaving += 1
            # 标记链路属性为可用
            for it in range(len(event.m_workPath) - 1):
                self.m_resourceMap[event.m_workPath[it]][event.m_workPath[it + 1]][event.m_occupiedWave] = 1
                self.P_link[event.m_workPath[it]][event.m_workPath[it + 1]][event.m_occupiedWave] = 0
                self.m_skdMap[event.m_workPath[it]][event.m_workPath[it + 1]][event.m_occupiedWave] = 0
            # 删除当前事件
            self.m_pq.remove(event)
        return

    def resource_allocation(self, event, algorithm_type='ksp_ff'):
        """
        返回选用波长和芯--芯可以交换
        遍历过程更加麻烦，耗时多，当相较于不可以交换的情况，资源更多
        :param event: 待分配资源的事件
        :param algorithm_type: 算法类型，便于切换算法
        :return: wave 返回的为各路径上最开始被占用的波长 （即波长wave至wave+needWave-1的波长都会被占用）
        """
        m_resourceMap = self.m_resourceMap
        p_link = self.P_link
        network = self.network
        fiber_type = self.fiber_type
        sourceNode = event.m_sourceNode
        destNode = event.m_destNode
        k = 3
        waveList = self.available_channel
        needWave = 1
        modulationF = 'BPSK'
        p = event.P

        if algorithm_type == 'ksp_ff':
            wave = algorithm.ksp_ff(m_resourceMap, p_link, network, fiber_type, sourceNode, destNode, k, waveList, needWave, modulationF, p)
        else:
            raise ValueError(f"Undefined algorithm type.")

        return wave

    def show_OSNR(self):
        """
        评估信道的光信噪比
        :return: none
        """

        return

    def show_SKR(self):
        """
        评估信道的光信噪比
        :return: none
        """

        return

    def run_test(self):
        """
        返回选用波长和芯--芯可以交换
        遍历过程更加麻烦，耗时多，当相较于不可以交换的情况，资源更多
        :return: none
        """
        self.m_sumOfFailedService = 0
        self.m_nextServiceId = 0
        self.generateServiceEventPair(self.m_nextServiceId)
        self.m_nextServiceId += 1
        self.ServiceQuantity = 0
        t1 = 0                                      # 记录时隙的变量，每到达一个时隙加1
        block_rate = []
        serve_leave = []
        while t1 != self.Ts:                        # t<Ts, 总的Ts数为100
            event_first = self.m_pq[0]
            self.num_leaving = 0
            self.num_arrival = 0

            count = 0                               # count记录真正到达的业务数量
            while int(event_first.m_time) == t1:
                if event_first.m_eventType["Arrival"] == 1 and int(event_first.m_holdTime+event_first.m_time-t1) != 0:
                    count += 1
                self.dealWithEvent(event_first)
                event_first = self.m_pq[0]

            t1 += 1
            print('time slot', t1-1, 'ended------------')

            self.show_OSNR()
            self.show_SKR()

            serve_leave.append(self.num_leaving)
            block_rate.append(self.m_sumOfFailedService / self.ServiceQuantity)
            print('block rate', self.m_sumOfFailedService / self.ServiceQuantity)
            print("current service amount:", len(self.m_pq))
            self.c_s_amount.append(len(self.m_pq))

        print('---------Summary of this run---------')
        print('Max block rate', max(block_rate))
        print('Average block rate', self.m_sumOfFailedService / self.ServiceQuantity)


if __name__ == '__main__':
    c = ClassicalService(TOPOLOGY=topology.net1, WaveNumber=16, Ts=200, lambda1=300, rou=0.5)
    c.run_test()
