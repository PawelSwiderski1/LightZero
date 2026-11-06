import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def action_name(action: Any) -> str:
    try:
        action = int(action)
    except (TypeError, ValueError):
        return str(action)
    return {0: "left", 1: "no-op", 2: "right"}.get(action, str(action))


def fmt_action(action: Any) -> str:
    try:
        return f"{int(action)}:{action_name(action)}"
    except (TypeError, ValueError):
        return str(action)


def fmt_path(path: Iterable[Any]) -> str:
    actions = list(path or [])
    if not actions:
        return "root"
    return ",".join(action_name(action) for action in actions)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(path: Path, phase: Optional[int], limit: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r") as f:
        for line in f:
            row = json.loads(line)
            if phase is not None and int(row.get("collection_phase", -1)) != int(phase):
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def print_root_actions(row: Dict[str, Any]) -> None:
    root_terms = row["trace"].get("intrinsic_root_diagnostics", [])
    root_terms = sorted(root_terms, key=lambda item: int(item.get("action", -1)))
    print("  root actions:")
    for item in root_terms:
        intrinsic_value = safe_float(item.get("intrinsic_value"))
        normalized_intrinsic_value = safe_float(item.get("normalized_intrinsic_value", intrinsic_value))
        reward_return = safe_float(item.get("intrinsic_reward_return_mean"))
        leaf_bootstrap = safe_float(item.get("intrinsic_leaf_value_contribution_mean"))
        print(
            "    "
            f"a{item.get('action')} {action_name(item.get('action')):<5} "
            f"visits {int(item.get('visit_count', 0)):>2} "
            f"prior {safe_float(item.get('prior')):>6.3f} "
            f"intrinsic {intrinsic_value:>7.3f} "
            f"norm {normalized_intrinsic_value:>5.3f} "
            f"reward_sum {reward_return:>7.3f} "
            f"leaf_boot {leaf_bootstrap:>7.3f} "
            f"score {safe_float(item.get('total_score')):>7.3f}"
        )


def print_simulations(row: Dict[str, Any], max_simulations: int) -> None:
    traces = row["trace"].get("tree", {}).get("simulation_traces", [])
    print(f"  simulations shown: {min(max_simulations, len(traces))}/{len(traces)}")
    for item in traces[:max_simulations]:
        print(
            "    "
            f"sim {int(item.get('simulation', -1)):>2} "
            f"path [{fmt_path(item.get('path_actions', []))}] "
            f"leaf_v {safe_float(item.get('leaf_intrinsic_value')):>7.3f} "
            f"backed {safe_float(item.get('backed_up_intrinsic_value')):>7.3f} "
            f"reward_sum {safe_float(item.get('backed_up_intrinsic_reward_return')):>7.3f} "
            f"leaf_boot {safe_float(item.get('backed_up_intrinsic_leaf_value_contribution')):>7.3f}"
        )


def print_tree_nodes(row: Dict[str, Any], max_nodes: int) -> None:
    nodes = row["trace"].get("tree", {}).get("nodes", [])
    nodes = sorted(nodes, key=lambda item: int(item.get("visit_count", 0)), reverse=True)
    print(f"  top visited nodes shown: {min(max_nodes, len(nodes))}/{len(nodes)}")
    for node in nodes[:max_nodes]:
        children = node.get("children", [])
        child_parts = []
        for child in sorted(children, key=lambda item: int(item.get("action", -1))):
            child_parts.append(
                f"a{child.get('action')}:n{int(child.get('visit_count', 0))}"
                f"/v{safe_float(child.get('intrinsic_value')):.2f}"
                f"/nv{safe_float(child.get('normalized_intrinsic_value', child.get('intrinsic_value'))):.2f}"
                f"/r{safe_float(child.get('intrinsic_reward_return_mean')):.2f}"
                f"/b{safe_float(child.get('intrinsic_leaf_value_contribution_mean')):.2f}"
            )
        print(
            "    "
            f"depth {int(node.get('depth', 0)):>1} "
            f"path [{fmt_path(node.get('path_actions', [])):<24}] "
            f"visits {int(node.get('visit_count', 0)):>2} "
            f"intrinsic {safe_float(node.get('intrinsic_value')):>7.3f} "
            f"norm {safe_float(node.get('normalized_intrinsic_value', node.get('intrinsic_value'))):>5.3f} "
            f"reward_sum {safe_float(node.get('intrinsic_reward_return_mean')):>7.3f} "
            f"leaf_boot {safe_float(node.get('intrinsic_leaf_value_contribution_mean')):>7.3f} "
            f"children {' '.join(child_parts)}"
        )


def print_row(row: Dict[str, Any], max_simulations: int, max_nodes: int) -> None:
    print("=" * 100)
    print(
        f"phase {row.get('collection_phase')} train_iter {row.get('train_iter_at_collect')} "
        f"env {row.get('env_id')} env_ep {row.get('env_episode_index')} "
        f"collect_ep {row.get('collection_episode_index')} t {row.get('timestep')}"
    )
    print(
        f"  obs pos {safe_float(row.get('obs_position')):.4f} "
        f"vel {safe_float(row.get('obs_velocity')):.4f} "
        f"selected {fmt_action(row.get('selected_action'))} "
        f"intrinsic_best {fmt_action(row.get('trace', {}).get('intrinsic_best_action'))}"
    )
    print(
        f"  future max pos 20/50/100/rem "
        f"{safe_float(row.get('future_max_position_20')):.3f} / "
        f"{safe_float(row.get('future_max_position_50')):.3f} / "
        f"{safe_float(row.get('future_max_position_100')):.3f} / "
        f"{safe_float(row.get('future_max_position_remaining')):.3f}"
    )
    print_root_actions(row)
    print_simulations(row, max_simulations)
    print_tree_nodes(row, max_nodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize intrinsic tree trace diagnostics JSONL.")
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-simulations", type=int, default=25)
    parser.add_argument("--max-nodes", type=int, default=16)
    args = parser.parse_args()

    rows = load_rows(args.jsonl_path, args.phase, args.limit)
    if not rows:
        print("No matching intrinsic tree traces found.")
        return
    for row in rows:
        print_row(row, args.max_simulations, args.max_nodes)


if __name__ == "__main__":
    main()
