import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TABLE_CELL_RE = re.compile(r"^\|(.+)\|$")
COLLECTOR_FIELD_RE = re.compile(r"^\s+([A-Za-z0-9_./-]+):\s+(.+?)\s*$")
BRACKET_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
LOGGER_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")


def parse_value(value: str) -> Any:
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def parse_timestamp(line: str) -> Optional[datetime]:
    match = BRACKET_TIMESTAMP_RE.match(line) or LOGGER_TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def split_table_cells(line: str) -> Optional[List[str]]:
    match = TABLE_CELL_RE.match(line.strip())
    if match is None:
        return None
    return [cell.strip() for cell in match.group(1).split("|")]


def parse_evaluator_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    lines = path.read_text(errors="replace").splitlines()
    idx = 0
    latest_timestamp = None
    while idx < len(lines):
        timestamp = parse_timestamp(lines[idx])
        if timestamp is not None:
            latest_timestamp = timestamp

        cells = split_table_cells(lines[idx])
        if cells is None or not cells or cells[0] != "Name":
            idx += 1
            continue

        value_cells = None
        search_idx = idx + 1
        while search_idx < min(idx + 5, len(lines)):
            possible_value_cells = split_table_cells(lines[search_idx])
            if possible_value_cells and possible_value_cells[0] == "Value":
                value_cells = possible_value_cells
                break
            search_idx += 1

        if value_cells is None:
            idx += 1
            continue

        names = cells[1:]
        values = value_cells[1:]
        table_record = {name: parse_value(value) for name, value in zip(names, values)}

        if "train_iter" in table_record and current:
            records.append(current)
            current = {}
        if "train_iter" in table_record:
            current["timestamp"] = latest_timestamp
        current.update(table_record)
        idx = search_idx + 1

    if current:
        records.append(current)
    return records


def parse_collector_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in path.read_text(errors="replace").splitlines():
        if "Collector Training Summary:" in line:
            if current:
                records.append(current)
            current = {"timestamp": parse_timestamp(line)}
            continue

        if current is None:
            continue

        match = COLLECTOR_FIELD_RE.match(line)
        if match is None:
            if current:
                records.append(current)
                current = None
            continue
        current[match.group(1)] = parse_value(match.group(2))

    if current:
        records.append(current)
    return records


def run_label(run_dir: Path) -> str:
    return run_dir.name


def eval_mean(record: Dict[str, Any]) -> Optional[float]:
    value = record.get("eval_episode_return_mean", record.get("reward_mean"))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def first_iter_at(records: List[Dict[str, Any]], threshold: float) -> str:
    for record in records:
        mean = eval_mean(record)
        if mean is not None and mean >= threshold:
            return str(record.get("train_iter", ""))
    return "-"


def first_envstep_at(records: List[Dict[str, Any]], threshold: float) -> str:
    for record in records:
        mean = eval_mean(record)
        if mean is not None and mean >= threshold:
            return fmt(record.get("train_envsteps", 0))
    return "-"


def first_runtime_min_at(records: List[Dict[str, Any]], threshold: float) -> str:
    for record in records:
        mean = eval_mean(record)
        if mean is not None and mean >= threshold:
            return fmt(seconds_to_minutes(record.get("runtime_sec")))
    return "-"


def seconds_to_minutes(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value) / 60.0
    return None


def estimate_other_runtime_min(summary: Dict[str, Any]) -> Optional[float]:
    values = [
        summary.get("final_runtime_min"),
        summary.get("collector_runtime_min"),
        summary.get("eval_runtime_min"),
    ]
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    # This is a coarse estimate because wall time starts at the first eval event.
    # The residual is mostly learner updates, replay sampling, logging, and checkpointing.
    return max(0.0, values[0] - values[1] - values[2])


def ms_per_iter(runtime_min: Any, final_iter: Any) -> Optional[float]:
    if not isinstance(runtime_min, (int, float)) or not isinstance(final_iter, (int, float)) or final_iter <= 0:
        return None
    return runtime_min * 60_000.0 / final_iter


def attach_train_envsteps_by_order(eval_records: List[Dict[str, Any]], collector_records: List[Dict[str, Any]]) -> None:
    for idx, eval_record in enumerate(eval_records):
        if idx == 0:
            eval_record["train_envsteps"] = 0
        elif collector_records:
            collector_idx = min(idx - 1, len(collector_records) - 1)
            eval_record["train_envsteps"] = collector_records[collector_idx].get("total_envstep_count")


def attach_runtime_from_timestamps(eval_records: List[Dict[str, Any]]) -> None:
    timestamped_records = [
        record for record in eval_records
        if isinstance(record.get("timestamp"), datetime)
    ]
    if not timestamped_records:
        return
    start_time = timestamped_records[0]["timestamp"]
    for record in timestamped_records:
        record["runtime_sec"] = (record["timestamp"] - start_time).total_seconds()


def load_scalar_events(run_dir: Path, tag: str):
    event_dir = run_dir / "log" / "serial"
    if not event_dir.exists():
        return []
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return []

    accumulator = event_accumulator.EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    try:
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            return []
        return accumulator.Scalars(tag)
    except Exception:
        return []


def parse_evaluator_events(run_dir: Path) -> List[Dict[str, Any]]:
    iter_values = load_scalar_events(run_dir, "evaluator_iter/eval_episode_return_mean")
    step_values = load_scalar_events(run_dir, "evaluator_step/eval_episode_return_mean")
    if not iter_values:
        return []

    optional_tags = {
        "reward_std": load_scalar_events(run_dir, "evaluator_iter/reward_std"),
        "reward_max": load_scalar_events(run_dir, "evaluator_iter/reward_max"),
        "reward_min": load_scalar_events(run_dir, "evaluator_iter/reward_min"),
        "eval_envstep_count": load_scalar_events(run_dir, "evaluator_iter/envstep_count"),
        "evaluate_time": load_scalar_events(run_dir, "evaluator_iter/evaluate_time"),
    }

    records = []
    start_wall_time = iter_values[0].wall_time
    for idx, scalar in enumerate(iter_values):
        record = {
            "train_iter": scalar.step,
            "eval_episode_return_mean": scalar.value,
            "runtime_sec": scalar.wall_time - start_wall_time,
        }
        if idx < len(step_values):
            record["train_envsteps"] = step_values[idx].step
        for name, values in optional_tags.items():
            if idx < len(values):
                record[name] = values[idx].value
        records.append(record)
    return records


def parse_collector_events(run_dir: Path) -> List[Dict[str, Any]]:
    reward_mean = load_scalar_events(run_dir, "collector_step/reward_mean")
    if not reward_mean:
        return []

    total_episode_count = load_scalar_events(run_dir, "collector_step/total_episode_count")
    total_duration = load_scalar_events(run_dir, "collector_step/total_duration")
    collect_time = load_scalar_events(run_dir, "collector_step/collect_time")
    records = []
    for idx, scalar in enumerate(reward_mean):
        record = {
            "total_envstep_count": scalar.step,
            "reward_mean": scalar.value,
        }
        if idx < len(total_episode_count):
            record["total_episode_count"] = total_episode_count[idx].value
        if idx < len(total_duration):
            record["total_duration"] = total_duration[idx].value
        if idx < len(collect_time):
            record["collect_time"] = collect_time[idx].value
        records.append(record)
    return records


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}"
    return str(value)


def print_table(headers: List[str], rows: Iterable[List[Any]]) -> None:
    rendered_rows = [[fmt(cell) for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[idx]) for row in rendered_rows)) if rendered_rows else len(header)
        for idx, header in enumerate(headers)
    ]
    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "  ".join("-" * width for width in widths)
    print(header_line)
    print(sep_line)
    for row in rendered_rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    eval_records = parse_evaluator_events(run_dir)
    collector_records = parse_collector_events(run_dir)
    if not eval_records:
        eval_records = parse_evaluator_log(run_dir / "log" / "evaluator" / "evaluator_logger.txt")
    if not collector_records:
        collector_records = parse_collector_log(run_dir / "log" / "collector" / "collector_logger.txt")
    if eval_records and "train_envsteps" not in eval_records[-1]:
        attach_train_envsteps_by_order(eval_records, collector_records)
    if eval_records and "runtime_sec" not in eval_records[-1]:
        attach_runtime_from_timestamps(eval_records)

    final_eval = eval_records[-1] if eval_records else {}
    best_eval = max((eval_mean(record) or float("-inf") for record in eval_records), default=None)
    final_collect = collector_records[-1] if collector_records else {}
    total_eval_time = sum(
        record["evaluate_time"] for record in eval_records
        if isinstance(record.get("evaluate_time"), (int, float))
    )

    summary = {
        "label": run_label(run_dir),
        "run_dir": run_dir,
        "eval_records": eval_records,
        "collector_records": collector_records,
        "num_evals": len(eval_records),
        "final_iter": final_eval.get("train_iter"),
        "final_eval_envsteps": final_eval.get("train_envsteps"),
        "final_runtime_min": seconds_to_minutes(final_eval.get("runtime_sec")),
        "final_eval_mean": eval_mean(final_eval),
        "best_eval_mean": None if best_eval == float("-inf") else best_eval,
        "first_iter_150": first_iter_at(eval_records, 150),
        "first_envsteps_150": first_envstep_at(eval_records, 150),
        "first_runtime_150": first_runtime_min_at(eval_records, 150),
        "first_iter_200": first_iter_at(eval_records, 200),
        "first_envsteps_200": first_envstep_at(eval_records, 200),
        "first_runtime_200": first_runtime_min_at(eval_records, 200),
        "final_train_envsteps": final_collect.get("total_envstep_count"),
        "final_train_episodes": final_collect.get("total_episode_count"),
        "final_collect_reward": final_collect.get("reward_mean"),
        "collector_runtime_min": seconds_to_minutes(final_collect.get("total_duration")),
        "eval_runtime_min": seconds_to_minutes(total_eval_time),
    }
    summary["learner_other_min"] = estimate_other_runtime_min(summary)
    summary["learner_other_ms_per_iter"] = ms_per_iter(summary["learner_other_min"], summary["final_iter"])
    summary["wall_ms_per_iter"] = ms_per_iter(summary["final_runtime_min"], summary["final_iter"])
    return summary


def print_summary(summaries: List[Dict[str, Any]]) -> None:
    print("\nRun Summary")
    print_table(
        [
            "run",
            "evals",
            "final_iter",
            "final_eval_steps",
            "wall_min",
            "final_eval",
            "best_eval",
            "first_150_iter",
            "first_150_steps",
            "first_150_min",
            "first_200_iter",
            "first_200_steps",
            "first_200_min",
            "train_envsteps",
            "train_episodes",
            "last_collect_reward",
            "collector_min",
            "eval_min",
            "learner+other_min",
            "learner+other_ms/iter",
            "wall_ms/iter",
        ],
        [
            [
                summary["label"],
                summary["num_evals"],
                summary["final_iter"],
                summary["final_eval_envsteps"],
                summary["final_runtime_min"],
                summary["final_eval_mean"],
                summary["best_eval_mean"],
                summary["first_iter_150"],
                summary["first_envsteps_150"],
                summary["first_runtime_150"],
                summary["first_iter_200"],
                summary["first_envsteps_200"],
                summary["first_runtime_200"],
                summary["final_train_envsteps"],
                summary["final_train_episodes"],
                summary["final_collect_reward"],
                summary["collector_runtime_min"],
                summary["eval_runtime_min"],
                summary["learner_other_min"],
                summary["learner_other_ms_per_iter"],
                summary["wall_ms_per_iter"],
            ]
            for summary in summaries
        ],
    )


def print_eval_curve(summaries: List[Dict[str, Any]], last: Optional[int]) -> None:
    run_maps = []
    train_iters = set()
    for summary in summaries:
        run_map = {}
        for record in summary["eval_records"]:
            train_iter = record.get("train_iter")
            mean = eval_mean(record)
            if train_iter is not None and mean is not None:
                run_map[train_iter] = mean
                train_iters.add(train_iter)
        run_maps.append(run_map)

    sorted_iters = sorted(train_iters)
    if last is not None:
        sorted_iters = sorted_iters[-last:]

    print("\nEval Mean By Train Iter")
    print_table(
        ["train_iter"] + [summary["label"] for summary in summaries],
        [[train_iter] + [run_map.get(train_iter) for run_map in run_maps] for train_iter in sorted_iters],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize CartPole ensemble MuZero run logs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories under data_ensemble_muzero.")
    parser.add_argument("--last", type=int, default=None, help="Only show the last N eval rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summaries = [summarize_run(run_dir) for run_dir in args.run_dirs]
    print_summary(summaries)
    print_eval_curve(summaries, args.last)


if __name__ == "__main__":
    main()
