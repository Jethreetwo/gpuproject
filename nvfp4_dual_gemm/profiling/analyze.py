from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to an Nsight Compute output file")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"File not found: {args.path}")

    print(f"Analyze: {args.path}")


if __name__ == "__main__":
    main()

