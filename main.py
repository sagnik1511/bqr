import argparse

from bqr.sim import simulator


def define_parser():

    parser = argparse.ArgumentParser("Trading Environment Simulator")

    # Add arguments to the parser
    parser.add_argument(
        "--policy_name", type=str, required=True, help="Name of the policy to use."
    )

    parser.add_argument(
        "--feature_extractor",
        type=str,
        default=None,
        help="Feature extractor class name. Must be a subclass of ActorCriticPolicy or None.",
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Size of the hidden layers in the neural network.",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to run during training.",
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default="ETHUSDT",
        help="Ticker symbol for the trading pair.",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="20240101",
        help="Start date for data retrieval in YYYYMMDD format.",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default="20240630",
        help="End date for data retrieval in YYYYMMDD format.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0 for no output, 1 for some output, 2 for full output.",
    )

    parser.add_argument(
        "--total_training_steps",
        type=int,
        default=100000,
        help="Total number of training steps to perform.",
    )

    return parser


if __name__ == "__main__":

    parser = define_parser()
    args = parser.parse_args()

    # Accessing parsed arguments
    print(f"Policy Name: {args.policy_name}")
    print(f"Feature Extractor: {args.feature_extractor}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Number of Episodes: {args.num_episodes}")
    print(f"Ticker: {args.ticker}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Verbose: {args.verbose}")
    print(f"Total Training Steps: {args.total_training_steps}")

    sim = simulator.Simulator(
        args.policy_name,
        args.feature_extractor,
        args.hidden_size,
        args.num_episodes,
        args.ticker,
        args.start_date,
        args.end_date,
        args.verbose,
        args.total_training_steps,
    )
    sim.simulate()
