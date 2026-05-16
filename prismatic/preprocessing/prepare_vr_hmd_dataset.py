"""
Convert Unity HMD trajectory recordings into an OmniVLA-friendly JSONL manifest.

The Unity recorder writes one session directory containing:
  - meta.json
  - trajectory.jsonl
  - images/frame_XXXXXX.png

This script scans those sessions and emits:
  - dataset_info.json
  - train.jsonl
  - val.jsonl

Each JSONL row stores absolute image paths plus precomputed navigation targets so the
PyTorch dataset can stay lightweight.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def local_delta_xz(
    current_x: float,
    current_z: float,
    current_yaw_rad: float,
    target_x: float,
    target_z: float,
) -> tuple[float, float]:
    """Project a world-space XZ delta into the agent's local forward-right frame."""
    dx = target_x - current_x
    dz = target_z - current_z

    forward = math.sin(current_yaw_rad) * dx + math.cos(current_yaw_rad) * dz
    right = math.cos(current_yaw_rad) * dx - math.sin(current_yaw_rad) * dz
    return forward, right


@dataclass
class TrajectoryFrame:
    frame_index: int
    image_path: Path
    x: float
    y: float
    z: float
    yaw_rad: float


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_sessions(input_root: Path) -> list[Path]:
    if input_root.is_file():
        raise ValueError(f"Expected a directory, got file: {input_root}")

    meta_path = input_root / "meta.json"
    traj_path = input_root / "trajectory.jsonl"
    if meta_path.exists() and traj_path.exists():
        return [input_root]

    index_path = input_root / "trajectory_index.jsonl"
    if index_path.exists():
        sessions = []
        for row in load_jsonl(index_path):
            session_dir = Path(row["session_dir"])
            if session_dir.exists():
                sessions.append(session_dir)
        if sessions:
            return sorted(set(sessions))

    sessions = []
    for meta in input_root.rglob("meta.json"):
        session_dir = meta.parent
        if (session_dir / "trajectory.jsonl").exists():
            sessions.append(session_dir)
    return sorted(set(sessions))


def load_session_frames(session_dir: Path) -> tuple[dict, list[TrajectoryFrame]]:
    meta = load_json(session_dir / "meta.json")
    trajectory_rows = load_jsonl(session_dir / "trajectory.jsonl")

    frames: list[TrajectoryFrame] = []
    for row in trajectory_rows:
        pos = row["position_world"]
        rot = row.get("rotation_euler_world", {})
        image_path = (session_dir / row["image_path"]).resolve()

        frames.append(
            TrajectoryFrame(
                frame_index=int(row["frame_index"]),
                image_path=image_path,
                x=float(pos["x"]),
                y=float(pos["y"]),
                z=float(pos["z"]),
                yaw_rad=math.radians(float(rot.get("y", 0.0))),
            )
        )

    return meta, frames


def estimate_metric_spacing(sessions: Iterable[list[TrajectoryFrame]]) -> float:
    distances: list[float] = []
    for frames in sessions:
        for prev_frame, next_frame in zip(frames[:-1], frames[1:]):
            dist = math.hypot(next_frame.x - prev_frame.x, next_frame.z - prev_frame.z)
            if dist > 1e-6:
                distances.append(dist)

    if not distances:
        return 0.25

    return max(statistics.median(distances), 1e-3)


def build_sample(
    frames: list[TrajectoryFrame],
    current_idx: int,
    goal_idx: int,
    history_length: int,
    action_horizon: int,
    metric_spacing: float,
    instruction: str,
    episode_id: str,
) -> dict:
    current = frames[current_idx]
    goal = frames[goal_idx]

    history_start = current_idx - history_length + 1
    history_frames = frames[history_start : current_idx + 1]

    actions = []
    for offset in range(1, action_horizon + 1):
        future = frames[min(current_idx + offset, len(frames) - 1)]
        forward, right = local_delta_xz(current.x, current.z, current.yaw_rad, future.x, future.z)
        delta_yaw = wrap_angle_rad(future.yaw_rad - current.yaw_rad)
        actions.append(
            [
                forward / metric_spacing,
                right / metric_spacing,
                math.cos(delta_yaw),
                math.sin(delta_yaw),
            ]
        )

    goal_forward, goal_right = local_delta_xz(current.x, current.z, current.yaw_rad, goal.x, goal.z)
    goal_delta_yaw = wrap_angle_rad(goal.yaw_rad - current.yaw_rad)

    return {
        "episode_id": episode_id,
        "instruction": instruction,
        "history_images": [str(frame.image_path) for frame in history_frames],
        "current_image": str(current.image_path),
        "goal_image": str(goal.image_path),
        "current_index": current_idx,
        "goal_index": goal_idx,
        "temp_dist": float(goal_idx - current_idx),
        "metric_waypoint_spacing": metric_spacing,
        "actions": actions,
        "goal_pose": [
            goal_forward / metric_spacing,
            goal_right / metric_spacing,
            math.cos(goal_delta_yaw),
            math.sin(goal_delta_yaw),
        ],
        "obj_pose_norm": [
            goal_forward / metric_spacing,
            goal_right / metric_spacing,
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Raw Unity capture root or one session dir.")
    parser.add_argument("--output-root", type=Path, required=True, help="Where train/val manifests should be written.")
    parser.add_argument("--train-split", type=float, default=0.9, help="Fraction of sessions assigned to train.")
    parser.add_argument("--history-length", type=int, default=6, help="Number of context images per sample.")
    parser.add_argument("--action-horizon", type=int, default=8, help="Number of future waypoints per action chunk.")
    parser.add_argument("--goal-offset", type=int, default=8, help="Frames between current frame and goal frame.")
    parser.add_argument(
        "--metric-waypoint-spacing",
        type=float,
        default=None,
        help="Override normalization spacing. By default it is estimated from the recordings.",
    )
    args = parser.parse_args()

    session_dirs = discover_sessions(args.input_root.resolve())
    if not session_dirs:
        raise RuntimeError(f"No Unity trajectory sessions found under {args.input_root}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    loaded_sessions: list[tuple[str, dict, list[TrajectoryFrame]]] = []
    for session_dir in session_dirs:
        meta, frames = load_session_frames(session_dir)
        if len(frames) < max(args.history_length, args.action_horizon + 1):
            continue
        episode_id = session_dir.name
        loaded_sessions.append((episode_id, meta, frames))

    if not loaded_sessions:
        raise RuntimeError("Found sessions, but none had enough frames to form a training sample.")

    metric_spacing = (
        args.metric_waypoint_spacing
        if args.metric_waypoint_spacing is not None
        else estimate_metric_spacing(frames for _, _, frames in loaded_sessions)
    )

    split_index = int(len(loaded_sessions) * args.train_split)
    if len(loaded_sessions) > 1:
        split_index = min(max(split_index, 1), len(loaded_sessions) - 1)
    else:
        split_index = 1

    train_sessions = loaded_sessions[:split_index]
    val_sessions = loaded_sessions[split_index:]

    manifests = {
        "train": output_root / "train.jsonl",
        "val": output_root / "val.jsonl",
    }

    counts = {"train": 0, "val": 0}
    for split_name, split_sessions in (("train", train_sessions), ("val", val_sessions)):
        with manifests[split_name].open("w", encoding="utf-8") as f:
            for episode_id, meta, frames in split_sessions:
                goal_label = str(meta.get("goal_label", "")).strip()
                instruction = goal_label if goal_label else f"reach the goal from session {episode_id}"
                last_current_idx = len(frames) - args.action_horizon - 1
                for current_idx in range(args.history_length - 1, last_current_idx + 1):
                    goal_idx = min(current_idx + args.goal_offset, len(frames) - 1)
                    if goal_idx <= current_idx:
                        continue
                    sample = build_sample(
                        frames=frames,
                        current_idx=current_idx,
                        goal_idx=goal_idx,
                        history_length=args.history_length,
                        action_horizon=args.action_horizon,
                        metric_spacing=metric_spacing,
                        instruction=instruction,
                        episode_id=episode_id,
                    )
                    f.write(json.dumps(sample) + "\n")
                    counts[split_name] += 1

    dataset_info = {
        "raw_input_root": str(args.input_root.resolve()),
        "num_sessions": len(loaded_sessions),
        "num_train_sessions": len(train_sessions),
        "num_val_sessions": len(val_sessions),
        "num_train_samples": counts["train"],
        "num_val_samples": counts["val"],
        "history_length": args.history_length,
        "action_horizon": args.action_horizon,
        "goal_offset": args.goal_offset,
        "metric_waypoint_spacing": metric_spacing,
        "modalities_supported": {
            "pose_only": 4,
            "pose_and_image": 5,
            "image_only": 6,
            "language_only": 7,
            "language_and_pose": 8,
        },
    }
    with (output_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(json.dumps(dataset_info, indent=2))


if __name__ == "__main__":
    main()
