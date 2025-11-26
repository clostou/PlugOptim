import os
import time
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from ML.post import PlotAniNet
from ML.torch.utils import all_seed
from environment import PlugSplineEnv, PlugSplineSurrEnv
from agent import AgentConfig, PPO2, pretrain_sensor

# TODO: 使用全部历史训练数据构建代理模型，替换PlugSplineEnv的耗时CFD计算
# TODO: 智能体学习过慢，可能存在知识遗忘的问题


class RL:

    def __init__(self):
        self.agent: Optional[PPO2] = None
        self.env: Optional[PlugSplineEnv] = None

        self.seed = 10  # 随机种子
        self.use_surrogate = True  # 是否使用代理模型环境
        self.interface = True  # 是否动态显示训练过程
        self.device = "cuda"  # device to use
        self.train_eps = 2000  # 训练的回合数
        self.test_eps = 10  # 测试的回合数
        self.max_steps = 200  # 每个回合的最大步数
        self.eval_eps = 5  # 评估的回合数
        self.eval_per_episode = 50  # 评估的频率

        self.steps = []  # 记录所有回合的内迭代数
        self.rewards = []  # 记录所有回合的奖励
        self.best_agent = None
        self.steps_test = []
        self.rewards_test = []
        self.info_counter = {}  # 计数器，用于统计环境的退出状态

    def init(self):
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
        # 智能体设置
        agent_cfg = AgentConfig(
            observation_sample_n=self.env.observation_dim[1],
            observation_feature_n=self.env.observation_dim[0],
            action_n=self.env.action_dim[0],
            update_freq=50,
            gamma=0.9  # 0.99
        )
        self.agent = PPO2(agent_cfg)
        # 智能体权重初始化
        pretrain_sensor(self.env.work_path, self.agent)
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
            beta = 0.9
            smoothed_ep_reward = 0.
        else:
            monitor = None
        reset_init_params = {'seed': self.seed, 'options': {'epoch_count': len(self.steps)}}
        # 回合循环
        for i_ep in range(self.train_eps):
            ep_reward = 0  # 记录一回合内的奖励
            ep_step = 0
            while True:
                try:
                    state, info = self.env.reset(**reset_init_params)  # 重置环境，返回初始状态
                    reset_init_params = {}  # 初始化参数只使用一次
                except AssertionError:
                    print("环境初始化失败（计算未收敛），重试……")
                else:
                    print("=" * 32, "回合 %d" % i_ep, "=" * 32)
                    break
            for _ in range(self.max_steps):
                ep_step += 1
                action = self.agent.sample_action(state.T)[0]  # 选择动作（从-1~1缩放）
                next_state, reward, terminated, truncated, info = \
                    self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                done = terminated or truncated
                self.agent.push(state.T, action, reward, None, done)  # 保存transition
                state = next_state  # 更新下一个状态
                self.agent.update()  # 更新智能体
                ep_reward += reward  # 累加奖励
                # 更新计数器
                key = info.get('type', 'Normal')
                self.info_counter[key] = self.info_counter.get(key, 0) + 1
                if done:
                    break
            if monitor:
                monitor.add_i(0, i_ep, ep_reward)
                smoothed_ep_reward = beta * smoothed_ep_reward + (1. - beta) * ep_reward
                monitor.add_i(1, i_ep, smoothed_ep_reward)
                monitor.update()
            if (i_ep + 1) % self.eval_per_episode == 0:
                sum_eval_reward = 0
                for _ in range(self.eval_eps):
                    eval_ep_reward = 0
                    state, info = self.env.reset(seed=self.seed)
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
        print("完成训练！")
        # return self.best_agent, {'rewards': self.rewards, 'steps': self.steps}

    def test(self):
        """使用特定环境测试智能体"""
        print("开始测试！")
        for i_ep in range(self.test_eps):
            ep_reward = 0  # 记录一回合内的奖励
            ep_step = 0
            state, info = self.env.reset(seed=self.seed)  # 重置环境，返回初始状态
            for _ in range(self.max_steps):
                ep_step += 1
                action = self.agent.predict_action(state.T)[0]  # 选择动作
                next_state, reward, terminated, truncated, info = \
                    self.env.step(self.env.delta_sp_max * action)  # 更新环境，返回transition
                done = terminated or truncated
                state = next_state  # 更新下一个状态
                ep_reward += reward  # 累加奖励
                if done:
                    break
            self.steps_test.append(ep_step)
            self.rewards_test.append(ep_reward)
            print(f"回合：{i_ep + 1}/{self.test_eps}，奖励：{ep_reward:.2f}")
        print("完成测试")
        # return {'rewards': self.rewards_test, 'steps': self.steps_test}

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
            'steps_test': self.steps_test,
            'rewards_test': self.rewards_test
        }
        path = os.path.join(self.env.work_path,  "rl_%d.pth" % int(time.time()))
        torch.save(data, path)
        print("RL state has been saved to '%s'" % path)

    def load(self, path: str):
        """从本地文件读取智能体及训练过程数据"""
        data = torch.load(path)
        self.steps = data.get('steps', [])
        self.rewards = data.get('rewards', [])
        self.steps_test = data.get('steps_test', [])
        self.rewards_test = data.get('rewards_test', [])
        agent_state_dict = data.get('agent', None)
        if agent_state_dict:
            self.agent.load(agent_state_dict)

    @staticmethod
    def smooth(data, weight: float = 0.9):
        """用于平滑曲线，类似于Tensorboard中的smooth曲线"""
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    @staticmethod
    def rolling_std(data, width: int = 20, weight: float = 0.9):
        """基于时间序列的局部波动（局部标准差）计算“伪方差带”，用于表示曲线震荡程度"""
        stds = np.zeros_like(data)
        for i in range(len(data)):
            left = max(0, i - width // 2)
            right = min(len(data), i + width // 2)
            stds[i] = np.std(data[left: right])
        return RL.smooth(stds, weight=weight)

    def plot_reward(self, use_test: bool = True):
        """绘制奖励曲线"""
        if use_test:
            rewards = self.rewards_test
        else:
            rewards = self.rewards
        mean_smooth = np.array(self.smooth(rewards))
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
        ax.set_xlabel('episodes')
        ax.set_ylabel('reward')
        ax.grid(alpha=0.5)
        fig.show()

    def __del__(self):
        self.env.close()


if __name__ == '__main__':
    rl = RL()
    rl.init()
    # rl.load(r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/rl_1763804143.pth')
    rl.train()


