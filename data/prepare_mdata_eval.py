from __future__ import annotations

import argparse
import json
import os
from typing import Any


NEW_PREFIX = "mdata/yollava-data/"


def _rewrite_value(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("mdata/yollava-data/"):
            return value
        if value.startswith("YoLLaVA/yollava-data/"):
            return NEW_PREFIX + value.split("YoLLaVA/yollava-data/", 1)[1]
        if value.startswith("./yollava-data/"):
            return NEW_PREFIX + value.split("./yollava-data/", 1)[1]
        if value.startswith("yollava-data/"):
            return NEW_PREFIX + value.split("yollava-data/", 1)[1]
        return value
    if isinstance(value, list):
        return [_rewrite_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_value(val) for key, val in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mdata for M2A evaluation")
    parser.add_argument(
        "--input",
        default="mdata/multimodel_locomo_formated.json",
        help="Path to multimodel_locomo_formated.json",
    )
    parser.add_argument(
        "--output",
        default="dataset/eval_dataset.json",
        help="Output path for evaluator-ready dataset",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = [_rewrite_value(sample) for sample in data]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(converted)} samples -> {args.output}")


if __name__ == "__main__":
    main()
