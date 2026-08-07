#!/usr/bin/env python3
"""Incrementally classify PX4 ULogs into the canonical JSON log database.

The catalog performs one selective core-topic scan per changed file. It performs a
second selective detail-topic scan only for files classified as flights. Budgets
are checked before any ULog is parsed, and progress is emitted before each scan.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np

from uloglib import get_topic, load_topics, relative_time_s, state_intervals


INDEXER_VERSION = 4
SCHEMA_VERSION = 2
DEFAULT_MAX_NEW_FILES = 20
DEFAULT_MAX_SCAN_MIB = 512.0
DEFAULT_MAX_FILE_MIB = 64.0
MAX_TIMELINE_EVENTS = 200
MAX_DROPOUT_EVENTS = 100
MAX_MESSAGE_RECORDS = 200
MAX_PENDING_NAME_PREVIEW = 12

CORE_TOPICS = {
    "actuator_armed",
    "vehicle_status",
    "takeoff_status",
    "vehicle_land_detected",
    "mission_result",
    "navigator_mission_item",
    "estimator_status_flags",
}
DETAIL_TOPICS = {
    "vehicle_local_position",
    "vehicle_gps_position",
    "battery_status",
}

# Known PX4 navigation states used only for human-readable labels and coarse
# purpose classification. Preserve numeric values and label unknown states.
NAV_NAMES = {
    0: "manual",
    1: "altctl",
    2: "posctl",
    3: "mission",
    4: "loiter",
    5: "rtl",
    10: "acro",
    14: "offboard",
    15: "stabilized",
    18: "land",
    19: "takeoff",
}


class BudgetExceeded(RuntimeError):
    pass


def emit_progress(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), file=sys.stderr, flush=True)


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def pending_name_summary(paths: Iterable[Path]) -> dict[str, Any]:
    names = [path.name for path in paths]
    return {
        "pending_name_preview": names[:MAX_PENDING_NAME_PREVIEW],
        "pending_names_truncated": len(names) > MAX_PENDING_NAME_PREVIEW,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def finite_stats(values: Iterable[Any]) -> dict[str, float | int] | None:
    array = finite(values)
    if len(array) == 0:
        return None
    return {
        "samples": len(array),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def dataset(log, topic: str, instance: int = 0):
    return get_topic(log, topic, instance=instance, required=False)


def crop_mask(log, data: dict[str, np.ndarray], start_s: float, end_s: float) -> np.ndarray:
    time_s = relative_time_s(log, data)
    return np.isfinite(time_s) & (time_s >= start_s) & (time_s <= end_s)


def changed_values(log, data, field: str, limit: int = MAX_TIMELINE_EVENTS) -> dict[str, Any]:
    if data is None or field not in data or len(data[field]) == 0:
        return {"events": [], "total_events": 0, "truncated": False}
    values = np.asarray(data[field])
    changed = np.r_[True, values[1:] != values[:-1]]
    times = relative_time_s(log, data)[changed]
    all_events = [
        {"time_s": round(float(timestamp), 3), "value": json_safe(value)}
        for timestamp, value in zip(times, values[changed])
    ]
    return {
        "events": all_events[:limit],
        "total_events": len(all_events),
        "truncated": len(all_events) > limit,
    }


def bool_airborne(data, field: str, airborne_value: int | bool) -> bool:
    return bool(data is not None and field in data and np.any(np.asarray(data[field]) == airborne_value))


def true_interval(log, data, field: str, expected: int | bool) -> tuple[float, float] | None:
    if data is None or field not in data:
        return None
    time_s = relative_time_s(log, data)
    log_end_s = float(log.last_timestamp / 1e6 - log.start_timestamp / 1e6)
    intervals = state_intervals(
        time_s,
        data[field],
        expected,
        label=f"{field}={expected}",
        series_end_s=log_end_s,
    )
    if not intervals:
        return None
    first = intervals[0]
    return first.start_s, first.end_s


def detected_flight_window(log, armed, status, takeoff, land, flags) -> tuple[float, float] | None:
    armed_intervals = [
        true_interval(log, armed, "armed", 1),
        true_interval(log, status, "arming_state", 2),
    ]
    airborne_intervals = [
        true_interval(log, takeoff, "takeoff_state", 5),
        true_interval(log, flags, "cs_in_air", 1),
    ]
    armed_intervals = [interval for interval in armed_intervals if interval is not None]
    airborne_intervals = [interval for interval in airborne_intervals if interval is not None]
    # vehicle_land_detected can be sparse or briefly disagree with the stronger
    # takeoff/EKF in-air states. Use it only when neither stronger source exists.
    if not airborne_intervals:
        land_interval = true_interval(log, land, "landed", 0)
        if land_interval is not None:
            airborne_intervals.append(land_interval)
    intervals = armed_intervals + airborne_intervals
    starts: list[float] = []
    ends: list[float] = []
    for interval in intervals:
        starts.append(interval[0])
        ends.append(interval[1])
    if not starts or not ends:
        return None
    start_s, end_s = max(starts), min(ends)
    return (start_s, end_s) if end_s > start_s else None


def classify_flight(nav_values: list[int], is_flight: bool) -> str:
    if not is_flight:
        return "ground_session"
    if 3 in nav_values:
        return "mission_flight"
    if 14 in nav_values:
        return "offboard_flight"
    if any(value in nav_values for value in (2, 4)):
        return "position_control_flight"
    return "flight_other_or_unknown_mode"


def mission_summary(log, result, item) -> dict[str, Any] | None:
    if result is None and item is None:
        return None
    summary: dict[str, Any] = {}
    if result is not None:
        for output, field, aggregate in (
            ("mission_ids", "mission_id", "unique"),
            ("sequence_total_max", "seq_total", "max"),
            ("sequence_reached_max", "seq_reached", "max"),
            ("finished_observed", "finished", "any"),
            ("failure_observed", "failure", "any"),
            ("warning_observed", "warning", "any"),
        ):
            if field not in result:
                continue
            values = finite(result[field])
            if len(values) == 0:
                continue
            if aggregate == "unique":
                summary[output] = sorted({int(value) for value in values})
            elif aggregate == "max":
                summary[output] = int(np.max(values))
            else:
                summary[output] = bool(np.any(values != 0))
    if item is not None and "sequence_current" in item and "nav_cmd" in item:
        sequence = np.asarray(item["sequence_current"])
        command = np.asarray(item["nav_cmd"])
        changed = np.r_[True, (sequence[1:] != sequence[:-1]) | (command[1:] != command[:-1])]
        time_s = relative_time_s(log, item)[changed]
        all_events = [
            {
                "time_s": round(float(timestamp), 3),
                "sequence": int(seq),
                "nav_cmd": int(cmd),
            }
            for timestamp, seq, cmd in zip(time_s, sequence[changed], command[changed])
        ]
        summary["item_events"] = all_events[:MAX_TIMELINE_EVENTS]
        summary["item_event_count"] = len(all_events)
        summary["item_events_truncated"] = len(all_events) > MAX_TIMELINE_EVENTS
    return summary


def quality_summary(log) -> tuple[str, dict[str, Any]]:
    all_events = [
        {
            "time_s": round((item.timestamp - log.start_timestamp) / 1e6, 3),
            "duration_ms": int(item.duration),
        }
        for item in log.dropouts
    ]
    total_ms = sum(item["duration_ms"] for item in all_events)
    if getattr(log, "file_corruption", False):
        quality = "corrupt"
    elif len(all_events) > 100 or total_ms > 1000:
        quality = "dropout_severe"
    elif all_events:
        quality = "minor_dropouts"
    else:
        quality = "clean"
    return quality, {
        "count": len(all_events),
        "total_ms": total_ms,
        "max_ms": max((item["duration_ms"] for item in all_events), default=0),
        "events": all_events[:MAX_DROPOUT_EVENTS],
        "events_truncated": len(all_events) > MAX_DROPOUT_EVENTS,
    }


def log_messages(log) -> dict[str, Any]:
    messages = [
        {
            "time_s": round((item.timestamp - log.start_timestamp) / 1e6, 3),
            "level": int(item.log_level),
            "text": item.message.rstrip(),
        }
        for item in log.logged_messages
    ]
    return {
        "records": messages[:MAX_MESSAGE_RECORDS],
        "total_records": len(messages),
        "truncated": len(messages) > MAX_MESSAGE_RECORDS,
    }


def detail_metrics(log, window_s: tuple[float, float]) -> dict[str, Any]:
    start_s, end_s = window_s
    metrics: dict[str, Any] = {
        "window_s": [round(start_s, 3), round(end_s, 3)],
        "duration_s": round(end_s - start_s, 3),
    }
    local = dataset(log, "vehicle_local_position")
    if local is not None:
        mask = crop_mask(log, local, start_s, end_s)
        if "z" in local:
            metrics["height_up_m"] = finite_stats(-np.asarray(local["z"])[mask])
        if "vx" in local and "vy" in local:
            speed = np.hypot(np.asarray(local["vx"])[mask], np.asarray(local["vy"])[mask])
            metrics["horizontal_speed_m_s"] = finite_stats(speed)
        if "x" in local and "y" in local:
            x = np.asarray(local["x"], dtype=np.float64)[mask]
            y = np.asarray(local["y"], dtype=np.float64)[mask]
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) >= 2:
                metrics["horizontal_path_m"] = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
                metrics["horizontal_displacement_m"] = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))

    gps = dataset(log, "vehicle_gps_position")
    if gps is not None:
        mask = crop_mask(log, gps, start_s, end_s)
        if "fix_type" in gps:
            values = finite(np.asarray(gps["fix_type"])[mask])
            metrics["gps_fix_types"] = sorted({int(value) for value in values})
        if "satellites_used" in gps:
            metrics["gps_satellites"] = finite_stats(np.asarray(gps["satellites_used"])[mask])

    battery = dataset(log, "battery_status")
    if battery is not None:
        mask = crop_mask(log, battery, start_s, end_s)
        for output, field in (
            ("battery_voltage_v", "voltage_v"),
            ("battery_current_a", "current_a"),
            ("battery_remaining", "remaining"),
        ):
            if field in battery:
                metrics[output] = finite_stats(np.asarray(battery[field])[mask])
        if "discharged_mah" in battery:
            values = finite(np.asarray(battery["discharged_mah"])[mask])
            if len(values):
                metrics["battery_discharged_delta_mah"] = float(values[-1] - values[0])
    return metrics


def build_record(path: Path, detail_mode: str, position: int, total: int) -> dict[str, Any]:
    stat = path.stat()
    emit_progress(
        "core_scan_start",
        position=position,
        total=total,
        file=path.name,
        size_bytes=stat.st_size,
        retained_topics=sorted(CORE_TOPICS),
    )
    log = load_topics(path, CORE_TOPICS)
    armed = dataset(log, "actuator_armed")
    status = dataset(log, "vehicle_status")
    takeoff = dataset(log, "takeoff_status")
    land = dataset(log, "vehicle_land_detected")
    flags = dataset(log, "estimator_status_flags")
    result = dataset(log, "mission_result")
    item = dataset(log, "navigator_mission_item")

    armed_any = bool_airborne(armed, "armed", 1) or bool_airborne(status, "arming_state", 2)
    airborne_any = (
        bool_airborne(takeoff, "takeoff_state", 5)
        or bool_airborne(land, "landed", 0)
        or bool_airborne(flags, "cs_in_air", 1)
    )
    is_flight = armed_any and airborne_any
    window_s = detected_flight_window(log, armed, status, takeoff, land, flags) if is_flight else None
    nav_timeline = changed_values(log, status, "nav_state")
    nav_values = [int(event["value"]) for event in nav_timeline["events"]]
    classification = classify_flight(nav_values, is_flight)
    quality, dropouts = quality_summary(log)
    params = log.initial_parameters
    missing = sorted(topic for topic in CORE_TOPICS if dataset(log, topic) is None)

    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "name": path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "firmware": log.msg_info_dict.get("ver_sw"),
        "hardware": log.msg_info_dict.get("ver_hw"),
        "flight_uuid": json_safe(params.get("COM_FLIGHT_UUID")),
        "logging": {
            "mode": json_safe(params.get("SDLOG_MODE")),
            "profile": json_safe(params.get("SDLOG_PROFILE")),
        },
        "classification": classification,
        "is_flight": is_flight,
        "quality": quality,
        "missing_core_topics": missing,
        "loaded_span_s": round((log.last_timestamp - log.start_timestamp) / 1e6, 3),
        "flight_window_s": None if window_s is None else [float(window_s[0]), float(window_s[1])],
        "timeline": {
            "armed": changed_values(log, armed, "armed"),
            "arming_state": changed_values(log, status, "arming_state"),
            "nav_state": nav_timeline,
            "nav_state_names": [
                {
                    "time_s": event["time_s"],
                    "value": NAV_NAMES.get(int(event["value"]), f"nav_{int(event['value'])}"),
                    "numeric_value": int(event["value"]),
                }
                for event in nav_timeline["events"]
            ],
            "takeoff_state": changed_values(log, takeoff, "takeoff_state"),
            "landed": changed_values(log, land, "landed"),
        },
        "mission": mission_summary(log, result, item),
        "dropouts": dropouts,
        "messages": log_messages(log),
        "flight_metrics": None,
        "provenance": {
            "indexer_version": INDEXER_VERSION,
            "detail_mode": detail_mode,
            "core_topics": sorted(CORE_TOPICS),
            "scan_passes": 1,
            "estimated_bytes_scanned": stat.st_size,
            "nav_label_mapping": "known PX4 nav-state subset; numeric values preserved",
        },
    }

    if is_flight and window_s is not None and detail_mode == "flights":
        emit_progress(
            "detail_scan_start",
            position=position,
            total=total,
            file=path.name,
            size_bytes=stat.st_size,
            retained_topics=sorted(DETAIL_TOPICS),
        )
        detail = load_topics(path, DETAIL_TOPICS)
        record["flight_metrics"] = detail_metrics(detail, window_s)
        record["provenance"].update(
            {
                "detail_topics": sorted(DETAIL_TOPICS),
                "scan_passes": 2,
                "estimated_bytes_scanned": stat.st_size * 2,
            }
        )

    emit_progress(
        "file_complete",
        position=position,
        total=total,
        file=path.name,
        classification=classification,
        quality=quality,
        scan_passes=record["provenance"]["scan_passes"],
    )
    return record


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def save_outputs(records_by_name: dict[str, dict[str, Any]], output_dir: Path, source_dir: Path) -> None:
    records = sorted(records_by_name.values(), key=lambda record: natural_key(Path(record["name"])))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "indexer_version": INDEXER_VERSION,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source_dir.resolve()),
        "records": records,
    }
    atomic_write_text(
        output_dir / "flight_index.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def load_existing(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {record["name"]: record for record in payload.get("records", [])}


def resolve_selected(log_dir: Path, names: list[str] | None, pattern: str | None) -> list[Path]:
    if names is not None:
        paths = []
        for name in names:
            if Path(name).name != name:
                raise ValueError(f"--names accepts basenames only, got {name!r}")
            paths.append(log_dir / name)
    else:
        paths = list(log_dir.glob(pattern or "*.ulg"))
    paths = sorted({path.resolve() for path in paths}, key=natural_key)
    if not paths:
        raise ValueError("selection matched no ULog files")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"selected files do not exist: {missing}")
    wrong_suffix = [path.name for path in paths if path.suffix.lower() != ".ulg"]
    if wrong_suffix:
        raise ValueError(f"selection contains non-.ulg files: {wrong_suffix}")
    return paths


def cache_matches(record: dict[str, Any] | None, path: Path, detail_mode: str) -> bool:
    if not record:
        return False
    stat = path.stat()
    provenance = record.get("provenance", {})
    # Keep unchanged raw-log records stable across indexer upgrades; --rebuild opts in to replacement.
    return bool(
        record.get("path") == str(path.resolve())
        and record.get("size_bytes") == stat.st_size
        and record.get("mtime_ns") == stat.st_mtime_ns
        and provenance.get("detail_mode") == detail_mode
    )


def check_budget(
    pending: list[Path],
    detail_mode: str,
    max_new_files: int,
    max_scan_mib: float,
    max_file_mib: float,
) -> dict[str, Any]:
    mib = 1024 * 1024
    oversize = [path for path in pending if path.stat().st_size > max_file_mib * mib]
    estimated_passes = 2 if detail_mode == "flights" else 1
    estimated_upper_bytes = sum(path.stat().st_size * estimated_passes for path in pending)
    plan = {
        "pending_files": len(pending),
        **pending_name_summary(pending),
        "estimated_upper_scan_bytes": estimated_upper_bytes,
        "estimated_upper_scan_mib": round(estimated_upper_bytes / mib, 3),
        "detail_mode": detail_mode,
        "budget": {
            "max_new_files": max_new_files,
            "max_scan_mib": max_scan_mib,
            "max_file_mib": max_file_mib,
        },
    }
    if oversize:
        preview = ", ".join(
            f"{path.name} ({path.stat().st_size / mib:.1f} MiB)" for path in oversize[:MAX_PENDING_NAME_PREVIEW]
        )
        suffix = " ..." if len(oversize) > MAX_PENDING_NAME_PREVIEW else ""
        raise BudgetExceeded(
            f"{len(oversize)} file(s) exceed --max-file-mib={max_file_mib:g}: "
            + preview
            + suffix
        )
    if len(pending) > max_new_files:
        raise BudgetExceeded(f"pending files {len(pending)} exceed --max-new-files={max_new_files}")
    if estimated_upper_bytes > max_scan_mib * mib:
        raise BudgetExceeded(
            f"estimated upper scan {estimated_upper_bytes / mib:.1f} MiB exceeds --max-scan-mib={max_scan_mib:g}"
        )
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally classify selected PX4 ULogs with bounded selective scans."
    )
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--names", nargs="+", help="Exact .ulg basenames to consider")
    selection.add_argument("--glob", dest="pattern", help="Glob inside log_dir, for example 'log_8*.ulg'")
    parser.add_argument("--details", choices=("flights", "none"), default="flights")
    parser.add_argument("--max-new-files", type=int, default=DEFAULT_MAX_NEW_FILES)
    parser.add_argument("--max-scan-mib", type=float, default=DEFAULT_MAX_SCAN_MIB)
    parser.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    parser.add_argument("--rebuild", action="store_true", help="Ignore unchanged-file cache")
    parser.add_argument("--dry-run", action="store_true", help="Resolve selection and budgets without parsing ULogs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    log_dir = args.log_dir.resolve()
    output_dir = args.output.resolve()
    if not log_dir.is_dir():
        raise NotADirectoryError(log_dir)
    if args.max_new_files <= 0 or args.max_scan_mib <= 0 or args.max_file_mib <= 0:
        raise ValueError("scan budgets must be positive")

    selected = resolve_selected(log_dir, args.names, args.pattern)
    records_by_name = load_existing(output_dir / "flight_index.json")
    pending = [
        path
        for path in selected
        if args.rebuild or not cache_matches(records_by_name.get(path.name), path, args.details)
    ]
    skipped = [path.name for path in selected if path not in pending]
    try:
        plan = check_budget(
            pending,
            args.details,
            args.max_new_files,
            args.max_scan_mib,
            args.max_file_mib,
        )
    except BudgetExceeded as error:
        emit_progress(
            "budget_blocked",
            error=str(error),
            selected_files=len(selected),
            pending_files=len(pending),
            **pending_name_summary(pending),
        )
        return 3

    emit_progress(
        "plan",
        selected_files=len(selected),
        skipped_unchanged=len(skipped),
        **plan,
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "skipped": skipped, **plan}, ensure_ascii=False))
        return 0

    processed: list[str] = []
    for position, path in enumerate(pending, start=1):
        try:
            previous = records_by_name.get(path.name)
            record = build_record(path, args.details, position, len(pending))
            if previous is not None and "task_synopsis" in previous:
                record["task_synopsis"] = previous["task_synopsis"]
            records_by_name[path.name] = record
        except Exception as error:
            emit_progress(
                "file_error",
                position=position,
                total=len(pending),
                file=path.name,
                error_type=type(error).__name__,
                error=str(error),
                checkpointed_files=processed,
            )
            return 1
        processed.append(path.name)
        save_outputs(records_by_name, output_dir, log_dir)
        emit_progress(
            "checkpoint_saved",
            position=position,
            total=len(pending),
            file=path.name,
            index=str(output_dir / "flight_index.json"),
        )

    if not pending:
        save_outputs(records_by_name, output_dir, log_dir)
    result = {
        "processed": processed,
        "skipped_unchanged": skipped,
        "index": str(output_dir / "flight_index.json"),
        "classifications": {
            path.name: records_by_name[path.name]["classification"] for path in selected
        },
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
