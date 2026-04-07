import random
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import itertools
import numpy as np

from mcf_core import MultiCore
from calculate_reward import calculate_reward2
from core_code import cores_code
import topology
from event_process import poisson_event_queue

class QKDNetworkEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
            self,
            node_num=14,
            wave_num=16,
            core_num=7,
            topo=np.array(topology.topology1),
            ksp_index=1,
            ksp_weight="hop",
            erlang=50,
            target=('service', 5000),
            seed=53,
            allow_block=False,
            allow_mix_quantum_core=False
    ):
        super().__init__()

        # ----------------------------------------- #
        # 网络参数定义
        # ----------------------------------------- #
        self.node_num = node_num  # 网络节点数
        self.erlang = erlang  # 网络业务量
        self.topo = topo
        self.graph = nx.from_numpy_array(self.topo)
        self.ksp_index = ksp_index
        self.ksp_weight = ksp_weight
        self._link_path_list = self._k_shortest_path()

        self.wave_num = wave_num  # 波长数量 （波长编号范围 [0, wave_num-1]）
        self.core_num = core_num  # 纤芯数量 （纤芯编号范围 [0, core_num-1]）
        self._mcf_core = MultiCore(self.core_num)  # 定义多芯光纤的纤芯类，所有纤芯相关的处理都通过该内部对象解决
        self.c_core_list, self.q_core_list = (
            self._mcf_core.core_allocation())  # 分配经典纤芯、量子纤芯范围
        self._first_neighbor, self._secondary_neighbor, self._hij_matrix = cores_code(
            self.core_num)
        self._qkd_transceiver = np.zeros((self.node_num, self.node_num
                                          ), dtype=np.int32)

        # ----------------------------------------- #
        # 定义每个Episode结束方式（时隙数/请求数）
        # ----------------------------------------- #
        self.target_type, self.target_num = target
        self.env_seed = seed
        # 所有需要随机数的底方，
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
        _hold_time = 4
        # 在环境初始化时生成时间序列，避免重复运算
        self._strict_reach = True
        # self._event_queue 代表预先生成的事件列表，不用每次都重新生成
        # _max_id 代表到达的业务总数
        self._event_queue, _max_id = poisson_event_queue(
            _hold_time/self.erlang, _hold_time, target, topo, strict_reach=self._strict_reach, rng=self._rng)
        # _max_id 代表到达的业务总数
        # _event_queue_index 代表当前环境处理到第几条业务到达/离开事件
        self._event_queue_index = 0
        # _event_resource_list 代表事件占用的资源列表
        self._event_resource_list = [(None,None,None) for _ in range(_max_id)]

        # ----------------------------------------- #
        # 观察空间 Observation Space
        # ----------------------------------------- #
        # _request_map表示当前，因为需要包含决策所需的全部信息
        # 目前包含源节点、目的节点两个信息，后续可以增加维度
        self._request_map = np.zeros(2, dtype=np.int8)
        # _resource_map表示 QKD 网络光纤资源的占用情况
        # 四个维度分别代表：链路起点，链路终点，纤芯数量，波长数量；
        # 1为可用，0为不可用，2为正在被使用，3为量子信道。
        self._resource_map = np.zeros((node_num, node_num, core_num, wave_num), dtype=np.int8)
        if allow_mix_quantum_core:
            obs_dim = self.core_num + self.wave_num + 5
            obs_max = max(self.core_num * self.wave_num, self.node_num)
        else:
            obs_dim = len(self.c_core_list) + self.wave_num + 5
            obs_max = max(len(self.c_core_list) * self.wave_num, self.node_num)
        # 这里的 np.float32 是为了兼容算法框架，实际上只会取整数值
        # _get_obs的写法决定了obs只会是整数，所以可以放心
        self.observation_space = spaces.Box(
            low=0, high=obs_max, shape=(obs_dim,), dtype=np.float32
        )

        # ----------------------------------------- #
        # 动作空间 Action Space
        # ----------------------------------------- #
        # 选择FCP编号
        self.allow_block = allow_block
        self.allow_mix_quantum_core = allow_mix_quantum_core
        if self.allow_block:
            # 允许主动的阻塞动作
            self._n_actions = self.ksp_index * len(self.c_core_list) * self.wave_num + 1
        else:
            # 不允许主动的阻塞动作，只有所有动作都不合法才会阻塞
            self._n_actions = self.ksp_index * len(self.c_core_list) * self.wave_num
        # 定义动作空间大小
        self.action_space = spaces.Discrete(self._n_actions)
        # 记录最后的action_mask 用于方法get_action_mask
        self.training_mode = False
        self._last_action_mask = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # 回到事件列表的第一项
        self._event_queue_index = 0

        # 初始化每条链路的量子信道数
        self._qkd_transceiver = np.zeros((self.node_num, self.node_num
                                     ), dtype=np.int32)

        # 初始化_resource_map资源占用情况
        # 尽量不用for循环判断，效率很低
        # 这里通过多维度的掩码来实现

        # 链路掩码：链路需要存在
        # TODO:如果需要为前向/后向信号分区，这里需要修改！
        valid_link_mask = self.topo > 0  #(N, N), bool
        valid_link_mask = valid_link_mask[..., None, None]  #(N, N, 1, 1), bool

        # 纤芯类型掩码：判断第三维度是经典芯/量子芯
        # 经典芯如果分前后向的话，拆开就好
        is_classic_core = np.isin(np.arange(self.core_num), self.c_core_list)[None, None, :, None]  #(1, 1, C, 1), bool
        is_quantum_core = np.isin(np.arange(self.core_num), self.q_core_list)[None, None, :, None]  #(1, 1, C, 1), bool

        # 波长掩码
        wave = np.arange(self.wave_num)[None, None, None, :]  # (1,1,1,W), bool
        classic_by_wave = (wave >= self._qkd_transceiver[..., None, None])  # (N,N,1,W)
        quantum_by_wave = (wave < self._qkd_transceiver[..., None, None])  # (N,N,1,W)

        # 链路掩码、纤芯掩码和波长掩码组合得到最终的经典信号/量子信号掩码
        mask_classic = valid_link_mask & is_classic_core & classic_by_wave  # 有效链路上经典芯的“非量子波长”
        mask_quantum = valid_link_mask & is_quantum_core & quantum_by_wave  # 有效链路上量子芯的“量子波长”

        # 依据掩码结果初始化_resource_map
        self._resource_map.fill(0)
        # 经典信号置1
        self._resource_map[mask_classic] = 1
        # 量子信号置3
        self._resource_map[mask_quantum] = 3

        if self.allow_mix_quantum_core:
            # 允许量子纤芯混传经典信号
            mask_classical_in_quantum_core = valid_link_mask & is_quantum_core & classic_by_wave
            self._resource_map[mask_classical_in_quantum_core] = 1

        self._request_map[0], self._request_map[1] = (
            self._event_queue[0].sourceNode, self._event_queue[0].destNode)
        obs = self._get_obs()

        info = {}
        self.training_mode = getattr(self, "training_mode", False)
        if not self.training_mode:
            info["action_mask"] = self.action_mask_flat()
            self._last_action_mask = info.get("action_mask")
        return obs, info

    def step(self, action):
        """ Take the action to deal with the event, the go to the next arrival event
        """
        # get the action
        if self.allow_block and action == self._n_actions - 1:
            node_path, core_path, wave_num = None, None, None
        else:
            node_path_index = 0
            core_path_num = int(action // self.wave_num)
            wave_num = int(action % self.wave_num)

            # get the event
            _event = self._event_queue[self._event_queue_index]
            sourceNode = _event.sourceNode
            destNode = _event.destNode
            path_cost, node_path = self._link_path_list[sourceNode][destNode][node_path_index]
            node_path = np.array(node_path)
            core_path = self._get_core_path(node_path, core_path_num, wave_num)

        if self._do_event(node_path, core_path, wave_num):
            # _do_event 返回True 代表动作可行,动作会执行完
            reward = calculate_reward2(
                m_workPath=node_path,
                core_list=core_path,
                reg_wave=wave_num,
                QKD_transceiver=self._qkd_transceiver,
                first_neighbor=self._first_neighbor,
                m_resourceMap=self._resource_map,
                WaveNumber=self.wave_num,
                core_num=self.core_num,
                node_num=self.node_num,
            )
            _event = self._event_queue[self._event_queue_index]
            self._event_resource_list[_event.id] = node_path, core_path, wave_num
        else:
            # _do_event 返回False 代表动作不可行,动作根本不会执行
            reward = -3

        self._event_queue_index += 1
        # pop all leaving event, get the first arrival event
        while (self._event_queue_index < len(self._event_queue)
               and self._event_queue[self._event_queue_index].eventType == "Leave"):
            _event = self._event_queue[self._event_queue_index]
            node_path, core_path, wave_num = self._event_resource_list[_event.id]
            self._undo_event(node_path, core_path, wave_num)
            self._event_queue_index += 1

        # judge if the target is truncated
        terminated = False
        if self._event_queue_index == len(self._event_queue):
            truncated = True
            s = self._rng.integers(self.node_num)
            d = self._rng.integers(self.node_num)
            while s == d or (self._strict_reach and not nx.has_path(self.graph, s, d)):
                d = self._rng.integers(self.node_num)
            self._request_map = np.array([s, d])
        else:
            truncated = False
            _event = self._event_queue[self._event_queue_index]
            # update the request_map
            self._request_map = np.array([_event.sourceNode, _event.destNode])

        # get the next state
        next_obs = self._get_obs()

        info = {
            "success": 1 if reward != -3 else 0,
            "inst_reward": float(reward),
        }
        self.training_mode = getattr(self, "training_mode", False)
        if not self.training_mode:
            info["action_mask"] =  self.action_mask_flat(),
            self._last_action_mask = info.get("action_mask")
        return next_obs, reward, terminated, truncated, info

    def get_action_mask(self):
        return self._last_action_mask

    def render(self):
        pass

    def close(self):
        pass

    def _get_obs(self):
        """ Flatten the observation space into a 1D vector.
        :return:
        """
        s, d = self._request_map[0], self._request_map[1]
        _, node_path = self._link_path_list[s][d][0]
        node_path = np.asarray(node_path, dtype=np.int16)
        src = node_path[:-1]
        dst = node_path[1:]
        L = len(src)
        C = len(self.c_core_list)
        W = self.wave_num

        # 取路径上的资源子张量: (L, C, W)
        res_slice = self._resource_map[
            src[:, None, None], dst[:, None, None],
            np.asarray(self.c_core_list)[None, :, None],
            np.arange(W)[None, None, :]
        ]
        free = (res_slice == 1)  # True 表示该 hop/core/wave 可用

        # (1) 每个 (core,wave) 是否“全路径可用”：沿 L 取 AND → (C, W)
        cw_ok = free.all(axis=0)

        # (2) 每个 core 可用的 wave 数量 → (C,)
        core_capacity = cw_ok.sum(axis=1).astype(np.float32)

        # (3) 每个 wave 被多少条 core 支持 → (W,)
        wave_support = cw_ok.sum(axis=0).astype(np.float32)

        # (4) 路径级拥塞概况：每个 hop 的“可用 (core,wave) 数”，再取均值/最小值（2 个标量）
        hop_cap = free.reshape(L, -1).sum(axis=1).astype(np.float32)  # 每 hop 的可用格子数
        hop_cap_mean = np.array([hop_cap.mean()], dtype=np.float32)
        hop_cap_min = np.array([hop_cap.min()], dtype=np.float32)

        # (5) (s,d) 基本信息：源/宿节点编号、路径长度
        sd_feat = np.array([s, d, L], dtype=np.float32)

        obs = np.concatenate([core_capacity, wave_support, hop_cap_mean, hop_cap_min, sd_feat], axis=0)
        return obs.astype(np.float32, copy=False)

        # obs = np.concatenate([
        #     self._request_map.reshape(-1),
        #     self._resource_map.reshape(-1)
        # ], axis=0).astype(np.float32, copy=False)
        # return obs

    def _do_event(self, path, core_path, wave_num):
        if path is None or core_path is None or wave_num is None:
            return False
        else:
            wave_num = int(wave_num)
            core_path = np.asarray(core_path)
            path = np.asarray(path)

            # 多条链路的起点序列
            sourceNode = path[:-1]
            # 多条链路的终点序列
            destNode = path[1:]

            _resource_map_val = self._resource_map[sourceNode,destNode,core_path,wave_num]
            if not np.all(_resource_map_val == 1):
                return False

            # Use the vector method to improve the calculation speed
            self._resource_map[sourceNode,destNode,core_path,wave_num] = 2
            self._resource_map[destNode,sourceNode,core_path,wave_num] = 0

            # 这里可以添加其他条件，比如阻塞率、OSNR等等
            return True

    def _undo_event(self, path, core_path, wave_num):
        if path is None or core_path is None or wave_num is None:
            return
        else:
            wave_num = int(wave_num)
            core_path = np.asarray(core_path)
            path = np.asarray(path)

            # 多条链路的起点序列
            sourceNode = path[:-1]
            # 多条链路的终点序列
            destNode = path[1:]

            # Use the vector method to improve the calculation speed
            self._resource_map[sourceNode, destNode, core_path, wave_num] = 1
            self._resource_map[destNode, sourceNode, core_path, wave_num] = 1
            return

    def _k_shortest_path(self):
        result = [[[] for _ in range(self.node_num)] for _ in range(self.node_num)]
        for i in range(self.node_num):
            for j in range(self.node_num):
                weight = None
                if self.ksp_weight == 'distance':
                    weight = "weight"
                gen = nx.shortest_simple_paths(self.graph, i, j, weight=weight)
                results = []
                for path in itertools.islice(gen, self.ksp_index):
                    if self.ksp_weight == 'distance':
                        cost = nx.path_weight(self.graph, path, weight=weight)
                    else:
                        cost = len(path) - 1
                    result[i][j].append((cost, path))
        return result

    def _get_core_path(self, path, core_path_num, wave_num):
        # 对于波长 wave_num，检查整条路径是否都能找到一个可用纤芯
        # 找到就选择“每一跳的首个可用纤芯”
        if self.allow_mix_quantum_core:
            # 允许经典芯、量子芯混传
            candidates_core = np.array(self.c_core_list + self.q_core_list, dtype=np.int8)
        else:
            # 不允许经典芯、量子芯混传
            candidates_core = np.array(self.c_core_list, dtype=np.int8)

        path = np.asarray(path, dtype=np.int8)
        core_path_num = int(core_path_num)
        wave_num = int(wave_num)

        # 多条链路的起点序列
        sourceNode = path[:-1]
        # 多条链路的终点序列
        destNode = path[1:]
        # 链路资源可用掩码
        avail = self._resource_map[
                    sourceNode[:, None], destNode[:, None], candidates_core[None, :], wave_num
                ] == 1

        # 有某一跳所有候选 core 都不可用
        if not np.all(avail.any(axis=1)):
            return None

        # 对每一跳，取从 core_path_num 开始的第一个 True 的下标
        # 先把前面的 core 列置 False 实现“偏移”
        if core_path_num > 0:
            avail[:, :core_path_num] = False

        picked_idx = avail.argmax(axis=1)  # (L,)
        # 若某一跳全 False，argmax 会给 0，但上面已经 any 检过，不会发生
        return candidates_core[picked_idx].tolist()

    def action_mask_flat(self):
        s, d = self._request_map[0], self._request_map[1]
        _, k_path = self._link_path_list[s][d][0]
        k_path = np.asarray(k_path, dtype=np.int16)
        # 多条链路的起点序列
        src_list = k_path[:-1]
        # 多条链路的终点序列
        dst_list = k_path[1:]

        if self.allow_mix_quantum_core:
            # 允许经典芯、量子芯混传
            candidates_core = np.array(self.c_core_list + self.q_core_list, dtype=np.int8)
        else:
            # 不允许经典芯、量子芯混传
            candidates_core = np.array(self.c_core_list, dtype=np.int8)

        res_slice = self._resource_map[
            src_list[:, None, None],
            dst_list[:, None, None],
            candidates_core[None, :, None],
            np.arange(self.wave_num)[None, None, :]
        ]
        free = (res_slice == 1)
        # 沿 hop 维取 AND：每个(core,wave)是否全路径可用 → (C, W)
        full_ok = free.all(axis=0)
        # 后缀 OR：对每个w，检查下标 ≥ c的纤芯中是否有可用的
        suffix_any = np.logical_or.accumulate(full_ok[::-1, :], axis=0)[::-1, :]
        mask = suffix_any.reshape(-1)
        if self.allow_block:
            mask = np.concatenate([mask, np.array([True])], axis=0)
        return mask


if __name__ == '__main__':
    # from gymnasium.envs.registration import register
    #
    # register(
    #     id="MyEnv-v0",
    #     entry_point="path.to.module:MyEnv",
    #     max_episode_steps=100,
    #     # kwargs if needed
    # )

    node = 14
    if node == 14:
        topology_ = np.array(topology.topology1)
    elif node == 24:
        topology_ = np.array(topology.topology6)
    else:
        topology_ = np.ones((node, node))
    env = QKDNetworkEnv(seed=53, topo=topology_, allow_block=False)

    # 结构健检（会做空间/返回格式/边界检查）
    CHECK_ENV = True
    if CHECK_ENV:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)


    obs_, info_ = env.reset(seed=53)
    print("obs shape:", obs_.shape, "obs dtype:", obs_.dtype)

    total_reward, max_episode = 0, 5000
    for _ in range(max_episode):
        action_ = env.action_space.sample()
        obs_, reward_, terminated_, truncated_, info_ = env.step(action_)
        total_reward += reward_
        if terminated_ or truncated_:
            obs_, info_ = env.reset()
            total_reward = 0
    print(total_reward)
    env.close()

