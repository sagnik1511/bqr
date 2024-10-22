"""Trading Simulator for SPOT Trading"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tabulate import tabulate
import pandas as pd


class TradingSimulator(gym.Env):

    def __init__(self, data):
        super(TradingSimulator, self).__init__()
        self.data = data
        self.balance = 100000
        self.value = self.balance
        self.acquired_shares = 0
        self.curr_step = 0
        self.history = []

        self.action_space = spaces.Box(
            low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(7,), dtype=np.float32
        )

    def _get_observation(self):
        curr_info = self.data[self.curr_step]
        return np.array(
            [
                curr_info["open"],
                curr_info["high"],
                curr_info["low"],
                curr_info["close"],
                curr_info["vol"],
                self.balance,
                self.acquired_shares,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=0):
        np.random.seed(seed)
        self.balance = 100000
        self.value = self.balance
        self.acquired_shares = 0
        self.curr_step = 0

        return self._get_observation(), {}

    def step(self, action):
        act_direction = action[0]
        amount = action[1]

        curr_price = self.data[self.curr_step]["close"]

        # BUY Action
        if act_direction == 1 and self.balance > 0:
            invest_amount = self.balance * amount
            shares = invest_amount // curr_price
            self.balance -= shares * curr_price
            self.acquired_shares += shares

        # SELL Action
        elif act_direction == -1 and self.acquired_shares > 0:
            shares = int(self.acquired_shares * amount)
            balance_regained = shares * curr_price
            self.balance += balance_regained
            self.acquired_shares -= shares

        new_value = self.balance + (self.acquired_shares * curr_price)
        self.curr_step += 1

        reward = new_value - self.value
        self.value = new_value

        obs = self._get_observation()
        completed = self.curr_step >= len(self.data) - 1

        return obs, reward, completed, False, {}

    def render(self, mode="human"):
        curr_price = self.data[self.curr_step]["close"]
        self.history.append(
            [
                self.curr_step + 1,
                curr_price,
                self.balance,
                self.acquired_shares,
                self.value,
                self.value - 100000,
            ]
        )
        info = [
            ["Step", self.curr_step + 1],
            ["Price", curr_price],
            ["Balance", self.balance],
            ["Shares Acquired", self.acquired_shares],
            ["Portfolio Value", self.value],
            ["Gross Earnings", self.value - 100000],
        ]

        print(tabulate(info, tablefmt="grid"))

    def save_history(self):
        hist_df = pd.DataFrame(
            self.history,
            columns=[
                "step",
                "price",
                "balance",
                "shares_acquired",
                "portfolio_value",
                "gross_earnings",
            ],
        )
        hist_df.to_csv("simulation.csv", index=False)
