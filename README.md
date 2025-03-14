# PPO Automating Trading Bot for SPOT Market

## Overview
This project implements a Proximal Policy Optimization (PPO) trading bot for the ETH/USD spot market. It downloads and processes financial K-line (candlestick) data, applies reinforcement learning techniques, and evaluates the bot's performance across different configurations.

## Features
- **Automated Data Collection**: Downloads ETH/USD K-line data.
- **PPO Policy Implementation**: Uses reinforcement learning to train a trading agent.
- **Custom Trading Environment**: Designed using Gymnasium for realistic simulation.
- **Performance Evaluation**: Tested on various setups to compare profitability and risk.

## Installation
### Requirements
- Python 3.9+
- Gymnasium
- Stable-Baselines3
- Pandas
- NumPy
- Matplotlib

### Setup
1. Please use conda environment to install the packages
2. Clone the repository:
   ```bash
   git clone https://github.com/sagnik1511/bqr
   cd bqr/
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download or prepare K-line data and place it in the `data/` directory.

## Project Structure
```
.
├── README.md               # Project documentation
├── artifacts               # Stores trained models and related files
├── blotter.py              # Trading blotter for trade execution tracking
├── bqr                     # Core project module
│   ├── nn                  # Neural network components
│   │   ├── baseline.py     # Baseline ML models
│   ├── sim                 # Simulation module
│   │   ├── env.py          # Custom trading environment
│   │   ├── simulator.py    # Market simulator
│   └── utils               # Utility functions
│       ├── decoder.py      # Data decoding utilities
├── docs                    # Documentation
│   └── data.md             # Data-related documentation
├── logs                    # Log files for debugging and tracking
├── main.py                 # Entry point to run training/testing
├── notebooks               # Jupyter notebooks for analysis
│   └── explorations.ipynb  # Exploratory analysis notebook
├── requirements.txt        # List of dependencies
```

## Usage
### Running the Trading Simulation
To start a trading simulation using `main.py`, run the following command:
```bash
python main.py --policy_name my_policy --num_episodes 200 --ticker ETHUSDT
```
#### Command-line Arguments:
- `--policy_name` (str, required): Name of the trading policy to use.
- `--feature_extractor` (str, optional): Feature extractor class name (must be a subclass of `ActorCriticPolicy` or `None`).
- `--hidden_size` (int, default=64): Size of the hidden layers in the neural network.
- `--num_episodes` (int, default=100): Number of episodes for training.
- `--ticker` (str, default="ETHUSDT"): Ticker symbol for the trading pair.
- `--start_date` (str, default="20240101"): Start date for data retrieval (YYYYMMDD format).
- `--end_date` (str, default="20240630"): End date for data retrieval.
- `--verbose` (int, default=1): Verbosity level (0 = silent, 1 = standard, 2 = detailed).
- `--total_training_steps` (int, default=100000): Total number of training steps.

## Results
- Backtest results are saved in `results/`
- Performance plots are generated for analysis
- Logs and metrics are available in `logs/`

## Future Work
- Implementing alternative RL algorithms (e.g., DDPG, SAC)
- Enhancing feature engineering for better market predictions
- Enhancing the bot to simultaneously work on different assets.
- Deploying the bot on live exchanges with paper trading

## License
MIT License

## Created out of Curiosity!
