import argparse
from pathlib import Path

from .runner import Config, run


def main():
    parser = argparse.ArgumentParser(description="Run all CT-RAG synthetic baselines and ablations")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2024])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--traces", type=int, default=4)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--hop-decay", type=float, default=.7)
    args = parser.parse_args()
    try:
        config = Config(tuple(args.seeds), tuple(args.ks), args.traces,
                        args.dimensions, args.max_hops, args.hop_decay)
    except ValueError as error:
        parser.error(str(error))
    report = run(config, args.output)
    print(f"Wrote {len(report['results'])} query/K/baseline observations to {args.output}")


if __name__ == "__main__":
    main()
