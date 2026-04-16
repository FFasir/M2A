from __future__ import annotations

import argparse
import json
import os

# Must be set before importing modules that may load pymilvus/gRPC.
os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "60000")
os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "20000")
os.environ.setdefault("GRPC_HTTP2_MIN_PING_INTERVAL_WITHOUT_DATA_MS", "60000")
os.environ.setdefault("GRPC_HTTP2_BDP_PROBE", "0")

from agent.config import M2AConfig
from eval.evaluator import Evaluator
from eval.llm_judge import LLMJudge
from eval_wrapper import M2AEvaluationWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2A evaluation on a dataset file")
    parser.add_argument("--config", default="config.toml", help="Path to config TOML")
    parser.add_argument("--dataset_root", default="./dataset", help="Root directory containing eval file")
    parser.add_argument("--eval_file", default="eval_dataset.json", help="Eval file under dataset_root")
    parser.add_argument("--n_parallel", type=int, default=1, help="Number of parallel evaluators")
    parser.add_argument("--n_sample_conv", type=int, default=100, help="Number of conversations to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max QA samples per conversation")
    parser.add_argument("--n_sample_msg", type=int, default=10000, help="Max dialogue turns per conversation")
    parser.add_argument("--out", default="result.json", help="Output JSON path")
    parser.add_argument("--table_out", default="result_table.md", help="Markdown table output path")
    return parser.parse_args()


def _format_markdown_table(summary: dict) -> str:
    lines = []
    lines.append("| Split | Count | F1 | BLEU1 | LLM_JUDGE |")
    lines.append("|---|---:|---:|---:|---:|")
    overall = summary["overall"]
    lines.append(
        "| Overall | {count} | {f1:.4f} | {bleu:.4f} | {judge:.4f} |".format(
            count=summary["total_questions"],
            f1=overall.get("F1", 0.0),
            bleu=overall.get("BLEU1", 0.0),
            judge=overall.get("LLM_JUDGE", 0.0),
        )
    )
    for cat, data in summary["by_category"].items():
        metrics = data["metrics"]
        lines.append(
            "| {cat} | {count} | {f1:.4f} | {bleu:.4f} | {judge:.4f} |".format(
                cat=cat,
                count=data["count"],
                f1=metrics.get("F1", 0.0),
                bleu=metrics.get("BLEU1", 0.0),
                judge=metrics.get("LLM_JUDGE", 0.0),
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    config = M2AConfig.from_file_and_env(args.config)
    models = [M2AEvaluationWrapper(config) for _ in range(args.n_parallel)]

    judge = LLMJudge(
        base_url=os.environ.get("JUDGE_BASE_URL", config.llm.base_url),
        api_key=os.environ.get("JUDGE_API_KEY", config.llm.api_key),
        model=os.environ.get("JUDGE_MODEL", config.llm.model),
    )

    evaluator = Evaluator(models, judge, database_root_path=args.dataset_root)
    results = evaluator.evaluate_file(
        args.eval_file,
        max_samples=args.max_samples,
        n_sample_conv=args.n_sample_conv,
        n_sample_msg=args.n_sample_msg,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(args.table_out, "w", encoding="utf-8") as f:
        f.write(_format_markdown_table(results["summary"]))

    print(f"Saved results to {args.out}")
    print(f"Saved table to {args.table_out}")


if __name__ == "__main__":
    main()
