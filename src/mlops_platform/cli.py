from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlops-demand")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        from mlops_platform import __version__

        print(__version__)


if __name__ == "__main__":
    main()
