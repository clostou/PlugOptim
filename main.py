import os
import queue
import sys
import time
from copy import deepcopy
from collections import deque
from typing import Union, Callable, Optional

import pyDOE2

import numpy as np
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
import torch
from torch.multiprocessing import Process, Queue, set_start_method
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ML.post import PlotAniNet
from ML.torch.utils import all_seed
from ML.BOA import BOA
from ML.regress import RLS
from ML.reduce import PCA
from environment import PlugSplineEnv, PlugSplineSurrEnv
from agent import AgentConfig, PPO2, pretrain_sensor, pretrain_sensor_from_surr
from cfd_toolbox.utils import sec2str

# TODO: 使用全部历史训练数据构建代理模型，替换PlugSplineEnv的耗时CFD计算 ✔
# TODO: 智能体学习过慢，可能存在知识遗忘的问题，使用异步PPO ✔
# TODO: MultiRL评估时计算top10%reward ✔
# TODO: MultiRL收集帕累托前沿，用于后期fluent计算验证，并对帕累托点集进行降维（在代理模型的t-SNE平面上绘制动画）

# 代码分析
# cloc-2.08.exe F:\Nozzle\OpenFOAM\code F:\LGQ\python\cfd_toolbox F:\LGQ\python\ML --include-lang=Python --by-file
# pyreverse F:\Nozzle\OpenFOAM\code F:\LGQ\python\cfd_toolbox F:\LGQ\python\ML --output 'puml'  （dot依赖和Gaphor可视化）


class MaxValueDeque:
    """
    右进左出队列，并自动丢弃旧元素以保证最大值位于队列末端
    """

    def __init__(self, maxlen: int, sort: Callable = None):
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        self.sort = sort

    def _update(self):
        max_value = max(self.data, key=self.sort)
        for _ in range(self.data.index(max_value)):
            self.data.popleft()

    def popleft(self):
        value = self.data.popleft()
        self._update()
        return value

    def append(self, value):
        n = len(self.data)
        max_value_changed = True
        if n == 0:
            self.data.append(value)
        elif n == self.maxlen:
            self.data.append(value)
            self._update()
        else:
            if self.sort is None:
                if value >= self.data[0]:
                    self.data.clear()
                else:
                    max_value_changed = False
            else:
                if self.sort(value) >= self.sort(self.data[0]):
                    self.data.clear()
                else:
                    max_value_changed = False
            self.data.append(value)
        return max_value_changed

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class RL:
    """
    基于PlugSpline环境和PPO智能体的强化学习问题
    """

    def __init__(self, seed: int = 10, surr: bool = True, verbose: bool = True, **kwargs):
        self.agent: Optional[PPO2] = None
        self.env: Optional[Union[PlugSplineEnv, PlugSplineSurrEnv]] = None

        self.seed = seed  # 随机种子
        self.use_surrogate = surr  # 是否使用代理模型环境
        self.init_cf_min = None  # 环境的初始最小cf值
        self.init_spline_p = None  # 环境的初始构型
        self.interface = True  # 是否动态显示训练过程
        self.device = "cuda"  # device to use
        self.train_eps = 10000  # 训练的回合数
        self.test_eps = 100  # 测试的回合数
        self.max_steps = 200  # 每个回合的最大步数
        self.eval_eps = 30  # 评估的回合数
        self.eval_per_episode = 4000  # 评估的频率

        self.steps = []  # 记录所有回合的内迭代数
        self.rewards = []  # 记录所有回合的奖励
        self.rewards_eval = []  # 记录所有训练过程中的评估奖励
        self.info_counter = {}  # 计数器，用于统计环境的退出状态
        self.best_agent = None
        self.steps_test = []
        self.rewards_test = []
        self.performance = []  # 测试得到的性能指标（初始、最终）
        self.profile = []  # 测试得到的样条配置集合（初始、最终）
        self.best_profile = None

        self.verbose = verbose
        self._stdout = None

        self.__dict__.update(kwargs)

    def _verbose_trigger(self):
        if not self.verbose:
            # 调试用
            # self._out_write = sys.stdout.write
            # self._err_write = sys.stderr.write
            # sys.stdout.write = lambda s: self._out_write(f'[{self.seed}]' + s)
            # sys.stderr.write = lambda s: self._err_write(f'[{self.seed}]' + s)
            # 标准输出重定向至空输出
            self._stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def init(self, pretrain: bool = True):
        """初始化训练环境和智能体"""
        # 设置全局随机数种子
        all_seed(seed=self.seed)
        # RL环境设置
        env_base = PlugSplineSurrEnv if self.use_surrogate else PlugSplineEnv
        self.env = env_base(
            render_mode='human' if self.interface else None,
            thread_n=32,
            step_max=20
        )
        # 训练设置
        self.env.cf_baseline = 1.801
        self.init_cf_min = 1.5
        # self.init_spline_p = [0.99, 0.758, 0.727, 0.566, 0.513, 0.351, 0.225]
        # 智能体设置
        agent_cfg = AgentConfig(
            device=self.device,
            observation_sample_n=self.env.observation_dim[1],
            observation_feature_n=self.env.observation_dim[0],
            action_n=self.env.action_dim[0],
            update_freq=1024,
            gamma=0.8,  # 0.95
            k_epochs=4,
            batch_size=256,  # 512
            sensor_lr=1e-6,  # 5e-6
            actor_lr=3e-5,  # 3e-5
            critic_lr=3e-5,
            entropy_coef=-0.05,  # 0.001
            eps_clip=0.2  # 0.1
        )
        self.agent = PPO2(agent_cfg)
        # 智能体权重初始化
        if pretrain:
            # pretrain_sensor(self.env.work_path, self.agent)
            pretrain_sensor_from_surr(self.env.nozzle_surrogate_path, self.agent)
        pass

    def train(self):
        """使用特定环境训练智能体"""
        print("开始训练！")
        self.info_counter = {}  # 清空计数器
        best_ep_reward = 0  # 记录最大回合奖励
        if self.interface:
            monitor = PlotAniNet(title=f"Training curve on {self.device} of {self.agent.__class__.__name__}"
                                       f" for {self.env.__class__.__name__}",
                                 xlabel="episodes", ylabel="rewards", line_count=2, legend=["origin", "smoothed"])
            beta = 0.95
            smoothed_ep_reward = 0.
        else:
            monitor = None
        # 回合循环
        i_ep = 0
        seed = self.seed
        while i_ep < self.train_eps:
            ep_reward = 0  # 记录一回合内的奖励
            ep_step = 0
            while True:
                try:
                    # 重置环境，返回初始状态
                    state, info = self.env.reset(seed=seed, options={'epoch_count': len(self.steps),
                                                                     'cf_limit': self.init_cf_min,
                                                                     'action_integral': self.init_spline_p})
                    seed = None  # 仅首次重置状态时设置随机数种子
                except AssertionError:
                    print("环境初始化失败（计算未收敛），重试……")
                else:
                    print("=" * 32, "回合 %d" % i_ep, "=" * 32)
                    break
            skip_epoch = False
            for i_st in range(self.max_steps):
                ep_step += 1
                action = self.agent.sample_action(state.T)[0]  # 选择动作（从-1~1缩放）
                next_state, reward, terminated, truncated, info = \
                    self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                # 更新计数器
                key = info.get('type', 'Normal')
                self.info_counter[key] = self.info_counter.get(key, 0) + 1
                if i_st == 0 and terminated:  # 若第一步异常退出，则跳过该回合
                    skip_epoch = True
                    break
                done = terminated or truncated
                self.agent.push(state.T, action, reward, None, done)  # 保存transition
                state = next_state  # 更新下一个状态
                self.agent.update()  # 更新智能体
                ep_reward += reward  # 累加奖励
                if done:
                    break
            if skip_epoch:
                continue
            print(f"Reward: {ep_reward:.1f}")
            if monitor:
                monitor.add_i(0, i_ep, ep_reward)
                smoothed_ep_reward = beta * smoothed_ep_reward + (1. - beta) * ep_reward
                monitor.add_i(1, i_ep, smoothed_ep_reward)
                monitor.update()
            if (i_ep + 1) % self.eval_per_episode == 0:
                sum_eval_reward = 0
                reset_init_params = {'seed': self.seed, 'options': {'epoch_count': len(self.steps)}}
                for _ in range(self.eval_eps):
                    eval_ep_reward = 0
                    state, info = self.env.reset(**reset_init_params)
                    reset_init_params = {}
                    for _ in range(self.max_steps):
                        action = self.agent.predict_action(state.T)[0]  # 选择动作
                        next_state, reward, terminated, truncated, info = \
                            self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                        done = terminated or truncated
                        state = next_state  # 更新下一个状态
                        eval_ep_reward += reward  # 累加奖励
                        if done:
                            break
                    sum_eval_reward += eval_ep_reward
                mean_eval_reward = sum_eval_reward / self.eval_eps
                self.rewards_eval.append(mean_eval_reward)
                if mean_eval_reward >= best_ep_reward:
                    best_ep_reward = mean_eval_reward
                    self.best_agent = deepcopy(self.agent)
                    print(f"回合：{i_ep+1}/{self.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，",
                          f"最佳评估奖励：{best_ep_reward:.2f}，更新模型！", sep='')
                else:
                    print(f"回合：{i_ep+1}/{self.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，",
                          f"最佳评估奖励：{best_ep_reward:.2f}", sep='')
            self.steps.append(ep_step)
            self.rewards.append(ep_reward)
            i_ep += 1
        print("完成训练！")
        # return self.best_agent, {'rewards': self.rewards, 'steps': self.steps}

    def train_worker(self, queue1: Queue, queue2: Queue):
        """使用特定环境训练智能体"""
        self._verbose_trigger()
        self.info_counter = {}  # 清空计数器
        # 回合循环
        i_ep = 0
        seed = self.seed
        while i_ep < self.train_eps:
            ep_reward = 0  # 记录一回合内的奖励
            ep_step = 0
            state, info = self.env.reset(seed=seed, options={'cf_limit': self.init_cf_min,
                                                             'action_integral': self.init_spline_p})
            seed = None  # 仅首次重置状态时设置随机数种子
            for i_st in range(self.max_steps):
                ep_step += 1
                action = self.agent.sample_action(state.T)[0]  # 选择动作（从-1~1缩放）
                next_state, reward, terminated, truncated, info = \
                    self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                # 更新计数器
                key = info.get('type', 'Normal')
                self.info_counter[key] = self.info_counter.get(key, 0) + 1
                done = terminated or truncated
                self.agent.push(state.T, action, reward, None, done)  # 保存transition
                state = next_state  # 更新下一个状态
                ep_reward += reward  # 累加奖励
                if done:
                    break
            if len(self.agent.memory) >= self.agent.update_freq:
                # 发送transition
                queue1.put((self.seed, list(self.agent.memory.buffer)))  # 使用种子作为索引
                self.agent.memory.buffer.clear()
                self.agent.load(queue2.get())  # timeout=120
            self.steps.append(ep_step)
            self.rewards.append(ep_reward)
            i_ep += 1
        queue1.put((0, None))  # 退出信号
        time.sleep(1)

    def train_host(self, queue1: Queue, queue2: Queue, n_worker: int):
        """使用特定环境训练智能体"""
        self._verbose_trigger()
        print("开始训练！")
        t = time.time()
        best_ep_reward = MaxValueDeque(maxlen=20, sort=lambda x: x[0])  # 用于计算最大回合奖励并缓存最近几回合的智能体
        if self.interface:
            monitor = PlotAniNet(title=f"Training curve on {self.device} of {self.agent.__class__.__name__}"
                                       f" for {self.env.__class__.__name__}",
                                 xlabel="$Episodes$", ylabel="$Rewards$", line_count=3,
                                 legend=["$origin$", "$smoothed$", "$top10\\%$"])
            beta = 0.8
            smoothed_ep_reward = None
        else:
            monitor = None
        i_ep = 0  # 训练回合数（实际上为评估回合数）
        ep_step = 0  # 总步数
        exit_counter = 0
        caching = []
        while True:
            try:
                sig, data = queue1.get(timeout=60)
            except queue.Empty:
                print("Training process no response, exit...", file=sys.stderr)
                sig = 0
            if sig > 0:
                caching.append([sig, data])
                ep_step += len(data)
                if len(caching) >= n_worker:
                    caching.sort(key=lambda x: x[0])
                    for sig, data in caching:
                        self.agent.memory.buffer.extend(data)
                        queue2.put(self.agent.dump())
                    caching.clear()
                    self.agent.update()
                # ep_step += len(data)
                # self.agent.memory.buffer.extend(data)
                # self.agent.update()
                # queue2.put(self.agent.dump())
            else:
                exit_counter += 1
            if ep_step >= i_ep * self.eval_per_episode:
                i_ep += 1
                eval_reward = []
                reset_init_params = {'seed': self.seed,
                                     'options': {'epoch_count': len(self.steps),
                                                 'cf_limit': self.init_cf_min,
                                                 'action_integral': self.init_spline_p}}
                for _ in range(self.eval_eps):
                    eval_ep_reward = 0
                    state, info = self.env.reset(**reset_init_params)
                    reset_init_params = {'options': {'cf_limit': self.init_cf_min,
                                                     'action_integral': self.init_spline_p}}
                    for _ in range(self.max_steps):
                        action = self.agent.predict_action(state.T)[0]  # 选择动作
                        next_state, reward, terminated, truncated, info = \
                            self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                        done = terminated or truncated
                        state = next_state  # 更新下一个状态
                        eval_ep_reward += reward  # 累加奖励
                        if done:
                            break
                    eval_reward.append(eval_ep_reward)
                mean_eval_reward = np.mean(eval_reward)  # 计算评估奖励的平均值
                n_top10 = int(np.ceil(0.1 * self.eval_eps))
                upper_eval_reward = np.sort(eval_reward)[-n_top10:].mean()  # 计算评估奖励的top10%平均值
                # 计算最近几回合内的最佳奖励
                if best_ep_reward.append((mean_eval_reward, deepcopy(self.agent))):
                    self.best_agent = best_ep_reward[0][1]
                    print(f"回合：{i_ep}，评估奖励：{mean_eval_reward:.2f}，",
                          f"最佳评估奖励：{best_ep_reward[0][0]:.2f}，更新模型！", sep='')
                else:
                    print(f"回合：{i_ep}，评估奖励：{mean_eval_reward:.2f}，",
                          f"最佳评估奖励：{best_ep_reward[0][0]:.2f}", sep='')
                if monitor:
                    monitor.add_i(0, i_ep, mean_eval_reward)
                    if smoothed_ep_reward is None:
                        smoothed_ep_reward = mean_eval_reward
                    smoothed_ep_reward = beta * smoothed_ep_reward + (1. - beta) * mean_eval_reward
                    monitor.add_i(1, i_ep, smoothed_ep_reward)
                    monitor.add_i(2, i_ep, upper_eval_reward)
                    monitor.update()
                self.rewards_eval.append(mean_eval_reward)
                self.steps.append(ep_step)
            if exit_counter >= n_worker:
                break
        t = int(time.time() - t)
        print(f"总步数{ep_step:d}，耗时{sec2str(t):s}")
        print("完成训练！")

    def test(self, random: bool = True):
        """使用特定环境测试智能体（默认使用最优智能体）"""
        print("开始测试！")
        self.steps_test = []  # 清空之前的测试结果
        self.rewards_test = []
        self.performance = []
        self.profile = []
        agent = self.agent if self.best_agent is None else self.best_agent
        performance_max = 0.
        reset_init_params = {'seed': self.seed,
                             'options': {'epoch_count': len(self.steps),
                                         'cf_limit': self.init_cf_min,
                                         'action_integral': self.init_spline_p}}
        for i_ep in range(self.test_eps if random else 1):  # 若不随机测试，那么只需要进行一回合的测试
            ep_reward = 0  # 记录一回合内的奖励
            ep_step = 0
            state, info = self.env.reset(**reset_init_params)  # 重置环境，返回初始状态
            reset_init_params = {'options': {'cf_limit': self.init_cf_min,
                                             'action_integral': self.init_spline_p}}
            # 记录初始性能指标以及相应的塞式喷管参数
            performance_initial = self.env.cf_current
            profile_initial = self.env.action_integral  # self.env.state[:2]
            for _ in range(self.max_steps):
                ep_step += 1
                action = agent.predict_action(state.T)[0]  # 选择动作
                next_state, reward, terminated, truncated, info = \
                    self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                done = terminated or truncated
                state = next_state  # 更新下一个状态
                ep_reward += reward  # 累加奖励
                if done:
                    break
            self.steps_test.append(ep_step)
            self.rewards_test.append(ep_reward)
            # 记录性能指标以及最佳性能下的塞式喷管参数
            performance = self.env.cf_current
            profile = self.env.action_integral
            if performance > performance_max:
                performance_max = performance
                self.best_profile = profile
            self.performance.append([performance_initial, performance])
            self.profile.append([profile_initial, profile])
            print(f"回合：{i_ep + 1}/{self.test_eps}，奖励：{ep_reward:.2f}")
        reward_avg = np.mean(self.rewards_test)
        performance = np.array(self.performance)
        performance_avg = performance.mean(axis=0)
        performance_std = performance.std(axis=0)
        print(f"平均奖励：{reward_avg:.2f}")
        print(f"初始平均性能：{performance_avg[0]:.3f}±{performance_std[0]:.3f}",
              f"平均性能：{performance_avg[1]:.3f}±{performance_std[1]:.3f}",
              f"最佳性能：{performance_max:.3f}", sep='，')
        print("完成测试！")
        # return {'rewards': self.rewards_test, 'steps': self.steps_test}

    def evaluate(self, *initial_profile):
        """给定初始构型，运行智能体并记录轨迹"""
        print("开始运行！")
        rewards = []
        performances = []
        states = []
        for i_ep, init_p in enumerate(initial_profile):
            ep_reward = 0  # 记录一回合内的奖励
            ep_performance = []
            ep_state = []
            try:
                state, info = self.env.reset(options={'action_integral': init_p, 'cf_limit': self.init_cf_min})
            except ValueError:
                print("Calculation failed, skip point %s" % init_p)
            else:  # if np.all(self.env.action_integral == init_p):  # 确保初始型面为给定型面，而非随机选取
                ep_state.append(state)
                ep_performance.append(self.env.cf_current)
                for _ in range(self.max_steps):
                    action = self.agent.predict_action(state.T)[0]  # 选择动作
                    next_state, reward, terminated, truncated, info = \
                        self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                    if not terminated:
                        ep_state.append(state)
                        ep_performance.append(self.env.cf_current)
                    done = terminated or truncated
                    state = next_state  # 更新下一个状态
                    ep_reward += reward  # 累加奖励
                    if done:
                        break
                rewards.append(ep_reward)
                performances.append(ep_performance)
                states.append(ep_state)
            print(f"回合：{i_ep + 1}/{len(initial_profile)}，奖励：{ep_reward:.2f}")
        performance = [trace[-1] for trace in performances]
        print(f"平均奖励：{np.mean(rewards):.2f}")
        print(f"平均性能：{np.mean(performance):.3f}，最佳性能：{max(performance):.3f}")
        print(f"运行结束，返回{len(performance):d}/{len(initial_profile):d}个样本")
        return rewards, performances, states

    def save(self):
        """保存智能体及训练过程至本地文件"""
        if self.best_agent is None:
            agent = self.agent
        else:
            agent = self.best_agent
        data = {
            'agent': agent.dump(),
            'steps': self.steps,
            'rewards': self.rewards,
            'rewards_eval': self.rewards_eval,
            'info_counter': self.info_counter,
            'steps_test': self.steps_test,
            'rewards_test': self.rewards_test,
            'performance': self.performance,
            'profile': self.profile,
            'best_profile': self.best_profile
        }
        path = os.path.join(self.env.work_path,  "rl_%d.pth" % int(time.time()))
        torch.save(data, path)
        print("RL state has been saved to '%s'" % path)

    def load(self, path: str):
        """从本地文件读取智能体及训练过程数据（兼容模式）"""
        data = torch.load(path)
        self.steps = data.get('steps', [])
        self.rewards = data.get('rewards', [])
        self.rewards_eval = data.get('rewards_eval', [])
        self.info_counter = data.get('info_counter', {})
        self.steps_test = data.get('steps_test', [])
        self.rewards_test = data.get('rewards_test', [])
        self.performance = data.get('performance', [])
        self.profile = data.get('profile', [])
        self.best_profile = data.get('best_profile', None)
        agent_state_dict = data.get('agent', None)
        if agent_state_dict:
            self.agent.load(agent_state_dict, strict=False)

    @staticmethod
    def smooth(data, weight: float = 0.95, init_value: float = None):
        """用于平滑曲线，类似于Tensorboard中的smooth曲线"""
        last = data[0] if init_value is None else init_value
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    @staticmethod
    def rolling_std(data, weight: float = 0.95):
        """基于时间序列的局部波动（局部标准差）计算“伪方差带”，用于表示曲线震荡程度"""
        width = int(len(data) / 10)  # 自动计算卷积核宽度
        stds = np.zeros_like(data)
        for i in range(len(data)):
            left = max(0, i - width // 2)
            right = min(len(data), i + width // 2)
            stds[i] = np.std(data[left: right])
        return RL.smooth(stds, weight=weight)

    def plot_reward(self, test: bool = False):
        """绘制奖励曲线"""
        if test:
            rewards = self.rewards_test
        else:
            rewards = self.rewards_eval if len(self.rewards) == 0 else self.rewards
        mean_smooth = np.array(self.smooth(rewards, 0.))
        std_local = np.array(self.rolling_std(rewards))
        fig, ax = plt.subplots()
        fig.suptitle(f"RL reward curve")
        ax.plot(mean_smooth, linewidth=2)
        ax.fill_between(
            np.arange(len(rewards)),
            mean_smooth - std_local,
            mean_smooth + std_local,
            alpha=0.25
        )
        if not test and len(self.rewards) != 0:  # 绘制评估曲线
            ax.plot(np.arange(self.eval_per_episode, len(rewards) + self.eval_per_episode, self.eval_per_episode),
                    self.rewards_eval, '.-')
        ax.set_xlabel('episodes')
        ax.set_ylabel('rewards')
        ax.grid(alpha=0.5)
        fig.show()

    def __del__(self):
        self.env.close()
        if not self.verbose and self._stdout is not None:
            sys.stdout.close()
            sys.stdout = self._stdout


class MultiRL:

    def __init__(self, n_env: int = 4, seed: int = 10):
        self.n_env = n_env
        self.env_worker = [RL(seed=seed+i, surr=True, verbose=False, interface=False, device='cpu')
                           for i in range(n_env)]
        self.env_host = RL(seed=seed, surr=True, device='cpu')
        self.queue_trans = Queue()
        self.queue_param = Queue()
        self.process = [Process(target=env.train_worker, args=(self.queue_trans, self.queue_param))
                        for env in self.env_worker]

        self.base_path = None
        self.test_result = []
        self._trained = False

    def init(self, pretrain: bool = True, agent_path: str = None):
        self.env_host.init(pretrain=pretrain)
        self.base_path = self.env_host.env.work_path
        for env in self.env_worker:
            env.init(pretrain=pretrain)
        if agent_path is not None:
            self.env_host.load(agent_path)

    def train(self):
        for p in self.process:
            p.start()
        self.env_host.train_host(self.queue_trans, self.queue_param, self.n_env)
        self._trained = True

    def test(self, use_history: bool = False, verify_prop: float = 0.2, contour_n: int = None):
        """运行智能体测试，并验证结果
        1. use_history可选择读取历史测试结果来做验证
        2. 可指定验证比率verify_prop和计算云图的算例数counter_n（按排序后的结果从高往低计数）
        """
        # 运行测试
        if not (use_history and self.env_host.performance and self.env_host.profile):
            self.env_host.test(random=True)
        # 保存训练结果
        if self._trained:
            self.env_host.save()
        # 依次计算最优型面
        self.test_result = []
        env = PlugSplineEnv(
            work_path=os.path.join(self.base_path, 'cfd_optimal'),
            thread_n=32,
            cfd_continuity_limit=20,
            cfd_cf_limit=3
        )
        verify_n = int(min(max(verify_prop, 0.0), 1.0) * len(self.env_host.performance))
        indexes = np.argsort(np.array([item[1] for item in self.env_host.performance]))[: : -1]
        for i, ind in enumerate(indexes):
            cf_start, cf_end = self.env_host.performance[ind]
            _, spline_p = self.env_host.profile[ind]
            if i < verify_n:  # 验证计算时只筛选高C_f值的case
                try:
                    state, _ = env.reset(options={'action_integral': spline_p})
                except (ValueError, AssertionError) as e:  # 处理计算失败的情况
                    print(repr(e))
                    data = [cf_start, cf_end, 0., env.nozzle_dir_current]
                else:
                    data = [cf_start, cf_end, env.cf_current, env.nozzle_dir_current]
            else:
                data = [cf_start, cf_end, 0., '']
            self.test_result.append(data)
            print(*data)  # 输出优化前后的C_f值做对比
        # env.post_processing()
        # 信息统计
        results = tuple(zip(*self.test_result))
        indexes = np.lexsort([results[1], results[2]])[: : -1]
        cf_list = []
        err_list = []
        post_list = []
        if contour_n is None:
            contour_n = 0
        else:
            post_list.append(env.nozzle_target.base_path)  # 添加对参考构型的后处理
        print('-' * 54)
        print(f"{'Ind.':<5} {'Cf_a':<8} {'Cf_b':<8} {'Cf':<8} {'Err. %':<8} {'Inc. %':<8}")
        for i, ind in enumerate(indexes):
            _cf, cf, cf_real, dir = self.test_result[ind]
            if i < contour_n:
                post_list.append((ind, dir))
            increase = 100 * (cf - _cf) / _cf
            if cf_real != 0:
                cf_list.append(cf_real)
                error = 100 * abs(cf_real - cf) / cf_real
                err_list.append(error)
                print(f"{i:>5d} {_cf:>8.3f} {cf:>8.3f} {cf_real:>8.3f} {error:>8.2f} {increase:>8.2f}")
            else:
                print(f"{i:>5d} {_cf:>8.3f} {cf:>8.3f} {'-':>8} {'-':>8} {increase:>8.2f}")
        print('-' * 54)
        print(f"验证样本数：{len(cf_list):d}/{len(self.env_host.performance):d}")
        if len(env.nozzle_target.data) == 1:
            cf_baseline = env.cf_baseline
        else:
            env.nozzle_target.thread = 1  # linux系统不支持多进程训练神经网络
            cf_baseline = env.nozzle_target.calc_cf(n_int=400, read_net=True)
        print(f"基线性能：{cf_baseline:.3f}")
        if len(cf_list) > 0:
            print(f"平均性能：{np.mean(cf_list):.3f}，最佳性能：{max(cf_list):.3f}")
            print(f"平均误差：{np.mean(err_list):.2f}%")
        # 使用cfdpost后处理
        for ind, dir in post_list:
            print(f"Processing case {ind}: {dir}")
            env.post_processing(dir)
        print("%d张云图绘制完成" % len(post_list))

    def reduce_dynamic(self, data_file: str = 'pretrain_data_101.pth', n: int = None, profile_only: bool = True):
        """在预训练数据上应用智能体，并在降维平面上对优化过程进行可视化
        参考 `environment.reduce_pretrain_data`"""
        # TODO: 绘图颜色范围可能会变化的问题
        # TODO: 使用代理模型采样数据集而不是预训练数据集（代理模型内含大量非单调型线），并根据cf和密度在投影平面上绘制云图
        # TODO: 代理模型拟合曲面不平滑，训练时考虑加入单调性损失或嵌入型面样条曲线 -> 后期基于代理的强化学习会有很强的偏好：优化仅限于单调子空间 ✔ ✘
        # 定义文件路径
        data_path = os.path.join(self.base_path, data_file)  # 喷管
        # 从文件中读取数据
        data = torch.load(data_path)
        states = data['state']
        if profile_only:  # 仅考虑构型，或同时考虑构型和流场
            states = np.array([state[:2].flatten() for state in states])
        else:
            states = np.array([state.flatten() for state in states])
        # states = (states - states.mean()) / states.std()
        config = data['config']
        assert len(config) == len(states), "Length of config and states mismatch"
        label = np.array(data['label'])
        assert len(states) == len(label), "Length of states and labels mismatch"
        if n is not None:
            states = states[:n]
            config = config[:n]
            label = label[:n]
        else:
            n = len(states)
        # 使用智能体优化型面
        # self.env_host.env.model_check = False  # 代理模型使用预训练数据集中的配置易生成大量非单调曲线（震荡），因此关闭模型检测提高接受率
        reward_traces, label_traces, states_traces = self.env_host.evaluate(*config)
        # 递归最小二乘
        rls = RLS(states.T, label)
        rls.train(cycle_count=3)
        states_p1 = np.vstack([rls.classify(states.T), label]).T
        # PCA线性降维
        pca = PCA(states.T)
        pca.train(ndim=2)
        states_p2 = pca.project(states.T).T
        # t-SNE非线性降维
        pca_ = PCA(states.T)
        pca_.train(ndim=20)  # 先基于PCA降至20维
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=50)  # 10 or 50 (perplexity: 5~50, less than sample_n)
        _states = pca_.project(states.T).T
        states_p3 = tsne.fit_transform(_states)  # manifold
        knn_x = KNeighborsRegressor(n_neighbors=100, weights='distance', metric='euclidean')  # 使用knn拟合tsne流形
        knn_y = KNeighborsRegressor(n_neighbors=100, weights='distance', metric='euclidean')
        knn_x.fit(_states, states_p3[:, 0])
        knn_y.fit(_states, states_p3[:, 1])
        knn = lambda x: np.vstack([knn_x.predict(x), knn_y.predict(x)]).T
        # 计算所有时间步在降维平面上的投影
        label_reduced = []
        states_reduced = []
        frame_n = 50
        for label_i, states_i in zip(label_traces, states_traces):
            label_i = np.array(label_i)
            if profile_only:  # 仅考虑构型，或同时考虑构型和流场
                states_i = np.array([state[:2].flatten() for state in states_i])
            else:
                states_i = np.array([state.flatten() for state in states_i])
            states_reduced_i = np.hstack([
                np.vstack([rls.classify(states_i.T), label_i]).T,
                pca.project(states_i.T).T,
                knn(pca_.project(states_i.T).T)
            ])
            if len(label_i) == 1:
                label_reduced.append(np.repeat(label_i, frame_n))
                states_reduced.append(np.repeat(states_reduced_i, frame_n, axis=0))
            else:
                kind = 'linear' if len(label_i) == 2 else 'quadratic'
                x, x_new = np.linspace(0, 1, len(label_i)), np.linspace(0, 1, frame_n)
                label_i_inp = interp1d(x, label_i, kind=kind)(x_new)
                states_reduced_i_inp = []
                for line in states_reduced_i.T:
                    states_reduced_i_inp.append(interp1d(x, line, kind=kind)(x_new))
                states_reduced_i_inp = np.vstack(states_reduced_i_inp).T
                label_reduced.append(label_i_inp)
                states_reduced.append(states_reduced_i_inp)
        point_n = len(label_reduced)
        label_reduced = np.array(label_reduced).swapaxes(0, 1)
        states_reduced = np.array(states_reduced).swapaxes(0, 1)
        # 绘制动画
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax = ax.flatten()
        hist_counts, hist_bins, hist_patches =\
            ax[-1].hist(label, bins=np.linspace(1.2, 1.8, min(int(0.2 * point_n), 10)),
                        alpha=0.75, color='blue', edgecolor='black')
        hist_upper = 2 * ax[-1].get_ylim()[1]
        text = ax[-1].text(1.2, 0.9 * hist_upper, s='', fontsize=16, color='gray', ha='left', va='top')
        ax[-1].set_ylim(0, hist_upper)
        kwargs = {'marker': 'o', 's': 14, 'c': label, 'cmap': plt.get_cmap('viridis')}  # seismic
        scatters = [ax[i].scatter(*s.T, **kwargs) for i, s in enumerate([states_p1, states_p2, states_p3])]

        def init():
            # 绘制label分布
            ax[-1].set_xlabel('Value')
            ax[-1].set_ylabel('Frequency')
            ax[-1].set_title('Frequency Distribution Histogram')
            ax[-1].grid(alpha=0.5)
            # 可视化样本点
            ax[0].set_title('RLS', fontsize=20)
            diag = [0.6, 1.8]
            ax[0].plot(diag, diag, '--', color='gray', alpha=0.6)
            ax[1].set_title('PCA', fontsize=20)
            ax[2].set_title('t-SNE', fontsize=20)
            for i in range(3):
                s = states_reduced[:, :, 2 * i: 2 * (i + 1)].T
                ax[i].set_xlim(s[0].min() - 0.1, s[0].max() + 0.1)
                ax[i].set_ylim(s[1].min() - 0.1, s[1].max() + 0.1)
                ax[i].set_xticklabels([''] * len(ax[i].get_xticklabels()))
                ax[i].set_yticklabels([''] * len(ax[i].get_xticklabels()))
                ax[i].grid(alpha=0.5)
            # # 设置初始空数据
            # for patch in hist_patches:
            #     patch.set_height(0)
            # for i in range(3):
            #     scatter.set_offsets(np.empty((0, 2)))  # 空点集
            #     scatter.set_array(np.array([]))  # 空颜色数组
            # for ax_i in ax:
            #     ax_i.autoscale_view()
            return *scatters, *hist_patches, text

        def update(frame):
            # 更新直方图
            counts, bins = np.histogram(label_reduced[frame], bins=hist_bins)
            for count, patch in zip(counts, hist_patches):
                patch.set_height(count)
            text.set_text("%.2f" % np.mean(label_reduced[frame]))
            # 更新散点图
            for i in range(3):
                scatters[i].set_offsets(states_reduced[frame, :, 2*i: 2*(i+1)])
                scatters[i].set_array(label_reduced[frame])
            return *scatters, *hist_patches, text

        anim = FuncAnimation(fig, update, frames=frame_n, interval=40,
                             init_func=init, blit=True)
        fig.tight_layout()
        plt.show()
        anim.save(os.path.join(self.base_path, 'reduce_dynamic.gif'), writer='pillow', fps=24)


def draw_contour(base_path: str, *sub_dirs: str, thread_n=8):
    """使用cfdpost处理指定的目录。这通常用于执行`MultiRL.test`获取最优构型后，获取该构型的云图"""
    env = PlugSplineEnv(
        work_path=base_path,
        thread_n=thread_n,
        cfd_continuity_limit=20,
        cfd_cf_limit=3
    )
    if sub_dirs:
        for p in sub_dirs:
            p = os.path.join(env.work_path, p)
            print("Processing '%s' ..." % p)
            env.post_processing(p)
    else:
        env.post_processing()


def test_pretrain_sensor(seed: int = 42):
    """对比随机初始化智能体和感知器预训练智能体的学习性能"""
    # RL单线程训练
    # rl = RL(seed=seed)  # 训练代理模型v2时随机种子10能比42收敛到更好的结果，v3由于存在微调阶段区别不大（训练耗时近1h）
    # rl.init(pretrain=True)
    # rl.train()
    # rl.plot_reward()
    # print(rl.info_counter)
    # plt.pause(1)
    # input()
    # rl.save()

    # MultiRL多线程训练
    # rl = MultiRL(seed=seed)
    # rl.init(pretrain=True)
    # # rl.init(pretrain=True, agent_path=r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/rl_1769499377.pth')  # 子进程会卡死在_generate_spline_plug函数处
    # rl.train()
    # plt.pause(1)
    # input()
    # rl.test(verify_prop=0.0)

    # MultiRL测试
    rl = MultiRL(n_env=0, seed=seed)
    rl.init(pretrain=False, agent_path=r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL_all/rl_1772871041.pth')
    rl.test(use_history=False, verify_prop=0.3)
    #rl.reduce_dynamic()


    pass


def test_boa_on_surrogate(seed: int = 42):
    """使用BOA直接在代理模型上优化喷管构型"""
    env = PlugSplineSurrEnv(
        render_mode='human',
        step_max=1000,
        delta_sp_max=1.0,
        model_check=False  # 对于代理模型环境，可关闭模型检查
    )
    state, _ = env.reset(seed=seed)
    action_n = env.action_dim[0]
    eps = 1e-8  # 防止越界的极小量

    def func(x: np.ndarray):
        y = []
        for x_i in x.T:
            action = x_i - env.action_integral
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            assert not done, info.get('type') + f" ({x_i})"
            y.append(env.cf_current)
        return np.array(y)

    init_p = pyDOE2.lhs(action_n, samples=10, criterion='center', random_state=seed)
    boa = BOA(func, init_p.T,
              m=1.2, p=0.6,  # p=0.5/0.6/0.7, 0.6最优
              upper=[1. - eps] * action_n,
              lower=[0.01 + eps] * action_n,
              plot=True)
    boa.plot3D(plot_target=False)  # 绘制原函数曲面会超出动作空间限制
    time.sleep(3)
    np.random.seed(seed)
    for i in range(10):
        x0 = 0.99 * np.random.rand(action_n) + 0.01
        boa.step(x0, precision=1e-3)
        boa.plot3D(plot_target=False)
        time.sleep(1)
    x_target = boa.search([0.5] * action_n, precision=1e-3)
    cf_target = func(x_target).item()
    print("Maximum of Cf: %f" % cf_target)


def test_optimal_of_surrogate():
    """评估代理优化结果的真实CFD性能"""
    # base_path = r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/'
    base_path = r'F:\Nozzle\OpenFOAM\FluentWithDL'
    # 四种奖励下的强化学习结果
    files = ['rl_1764312102.pth', 'rl_1764312807.pth', 'rl_1764312915.pth', 'rl_1764313060.pth']
    cf, profile = [], []
    for f in files:
        data = torch.load(os.path.join(base_path, f))
        cf.append([np.argmax(data['performance']), np.max(data['performance']), np.mean(data['performance'])])
        profile.append(data['best_profile'].tolist())  # 样条塞式喷管插值点数据
    profile = np.array(profile)
    for item in cf:
        print(*item)
    print(profile)

    # 依次计算最优型面
    cf_real = []
    env = PlugSplineEnv(
        work_path=os.path.join(base_path, 'cfd_optimal'),
        thread_n=32
    )
    for spline_p in profile:
        state, _ = env.reset(options={'action_integral': spline_p})
        cf_real.append([env.nozzle_dir_current, env.cf_current])
    # env.post_processing()
    for item in cf_real:
        print(*item)


if __name__ == '__main__':
    # set_start_method('spawn')

    # rl = RL()
    # rl.init()
    # # rl.load(r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/rl_1764250270.pth')
    # # rl.train()
    # rl.test()
    # rl.save()
    # test_optimal_of_surrogate()

    # test_boa_on_surrogate()

    # test_pretrain_sensor()

    draw_contour(r'F:\Nozzle\OpenFOAM\FluentWithDL_all\cfd_optimal', thread_n=8)


