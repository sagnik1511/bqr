from bqr.sim.env import TradingSimulator
from bqr.data.fetch_data import get_custom_data
import random


if __name__ == "__main__":

    data = get_custom_data("ETHUSDT", "20240101", "20240531")
    env = TradingSimulator(data)

    for _ in range(10):
        obs, _ = env.reset()

        done = False
        while not done:
            trade_act = random.choice([-1, 0, 1])
            amount_act = random.random()
            obs, reward, done, _ = env.step([trade_act, amount_act])
        env.render()
