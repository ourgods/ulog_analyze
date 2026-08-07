#!/usr/bin/env python3
"""Composable command-line access to PX4 ULog signals."""

from __future__ import annotations

import argparse
import ast
import csv
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np

from flight_catalog import CORE_TOPICS, build_record, quality_summary
from signal_tools import at_frequency, band_rms, transfer_function, welch_peak
from uloglib import (
    SignalSpec,
    align_signals,
    available_schemas,
    get_topic,
    load_topics,
    parameter_diff,
    read_header,
    relative_time_s,
    sampling_info,
    state_intervals,
    topic_instances,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ALIAS_FILE = SCRIPT_DIR.parent / "signal_aliases.yaml"
SOURCE_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_]\w*(?:@(?:\d+|primary))?\.[A-Za-z_]\w*(?:\[\d+\])?")
PRIMARY_INSTANCE = -1
DEFAULT_MAX_FILE_MIB = 64.0
LOW_STACK_RE = re.compile(
    r"\[(?P<source>[^\]]+)\]\s+(?P<task>\S+)\s+low on stack!\s+\((?P<bytes>\d+)\s+bytes left\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Expression:
    label: str
    source: str
    tree: ast.Expression
    references: tuple[str, ...]


def load_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("PyYAML is required to read signal aliases") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases = payload.get("aliases", payload)
    if not isinstance(aliases, dict):
        raise ValueError(f"aliases must be a mapping in {path}")
    result = {}
    for name, value in aliases.items():
        expression = value.get("expression") if isinstance(value, dict) else value
        if not isinstance(expression, str):
            raise ValueError(f"alias {name!r} must contain a string expression")
        result[str(name)] = expression
    return result


def check_file_size(path: Path, max_file_mib: float) -> None:
    if max_file_mib <= 0:
        raise ValueError("--max-file-mib must be positive")
    size_mib = path.stat().st_size / (1024 * 1024)
    if size_mib > max_file_mib:
        raise ValueError(
            f"{path.name} is {size_mib:.1f} MiB; raise --max-file-mib explicitly "
            f"(current limit {max_file_mib:g} MiB)"
        )


def expand_alias(value: str, aliases: dict[str, str]) -> str:
    if "=" not in value and value in aliases:
        return f"{value}={aliases[value]}"
    if "=" in value:
        label, expression = value.split("=", 1)
        expression = aliases.get(expression.strip(), expression)
        return f"{label.strip()}={expression}"
    return value


def parse_reference(reference: str) -> tuple[str, str, int]:
    if "." not in reference:
        raise ValueError(f"signal must be TOPIC[@INSTANCE].FIELD, got {reference!r}")
    topic_with_instance, field = reference.rsplit(".", 1)
    if "@" in topic_with_instance:
        topic, instance_text = topic_with_instance.rsplit("@", 1)
        if instance_text == "primary":
            instance = PRIMARY_INSTANCE
        elif instance_text.isdigit():
            instance = int(instance_text)
        else:
            raise ValueError(f"invalid topic instance in {reference!r}")
    else:
        topic, instance = topic_with_instance, 0
    if not topic or not field:
        raise ValueError(f"invalid signal reference {reference!r}")
    return topic, field, instance


def reference_has_explicit_instance(reference: str) -> bool:
    return "@" in reference.rsplit(".", 1)[0]


def ensure_unambiguous_instances(log: Any, specs: Iterable[SignalSpec]) -> None:
    ambiguous: dict[str, list[int]] = {}
    for spec in specs:
        if reference_has_explicit_instance(spec.label):
            continue
        instances = topic_instances(log, spec.topic)
        if len(instances) > 1:
            ambiguous[spec.topic] = instances
    if ambiguous:
        details = "; ".join(
            f"{topic} has instances {instances}; use {topic}@INSTANCE.FIELD"
            for topic, instances in sorted(ambiguous.items())
        )
        raise ValueError(f"ambiguous multi-instance signal: {details}")


def parse_expression(raw: str, aliases: dict[str, str]) -> Expression:
    raw = expand_alias(raw.strip(), aliases)
    if "=" in raw:
        label, source = raw.split("=", 1)
        label, source = label.strip(), source.strip()
        if not re.fullmatch(r"[A-Za-z_]\w*", label):
            raise ValueError(f"expression label must be an identifier, got {label!r}")
    else:
        label, source = raw, raw
    references = tuple(dict.fromkeys(SOURCE_RE.findall(source)))
    if not references:
        raise ValueError(f"expression {raw!r} contains no TOPIC.FIELD reference")
    rewritten = source
    names: dict[str, str] = {}
    for index, reference in enumerate(references):
        name = f"__signal_{index}"
        names[reference] = name
        rewritten = rewritten.replace(reference, name)
    tree = ast.parse(rewritten, mode="eval")
    validate_expression(tree)
    return Expression(label=label, source=source, tree=tree, references=references)


def validate_expression(node: ast.AST) -> None:
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
               ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load, ast.Mod)
    if not isinstance(node, allowed):
        raise ValueError(f"unsupported expression syntax: {type(node).__name__}")
    for child in ast.iter_child_nodes(node):
        validate_expression(child)


def eval_expression(node: ast.AST, values: dict[str, np.ndarray]) -> np.ndarray | float:
    if isinstance(node, ast.Expression):
        return eval_expression(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise KeyError(f"missing expression variable {node.id}")
        return values[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = eval_expression(node.operand, values)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
        left = eval_expression(node.left, values)
        right = eval_expression(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        return left % right
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def source_specs(
    signals: Iterable[str], expressions: Iterable[str], aliases: dict[str, str], interpolation: str
) -> tuple[list[SignalSpec], list[Expression]]:
    parsed = [parse_expression(item, aliases) for item in expressions]
    references: list[str] = []
    for item in signals:
        references.append(item)
    for item in parsed:
        references.extend(item.references)
    specs: list[SignalSpec] = []
    seen: set[str] = set()
    for reference in references:
        if reference in seen:
            continue
        seen.add(reference)
        topic, field, instance = parse_reference(reference)
        specs.append(SignalSpec(reference, topic, field, instance=instance, interpolation=interpolation))
    if not specs:
        raise ValueError("provide at least one -s signal or -e expression")
    return specs, parsed


def signal_window(path: Path, mode: str | None) -> tuple[float, float] | None:
    if mode is None:
        return None
    if mode == "airborne":
        # build_record emits progress events. Keep command stdout machine-readable.
        with redirect_stdout(sys.stderr):
            window = build_record(path, "none", 1, 1).get("flight_window_s")
        if not window:
            raise ValueError(f"no detected airborne window in {path.name}")
        return float(window[0]), float(window[1])
    core = load_topics(path, CORE_TOPICS)
    if mode == "armed":
        data = get_topic(core, "actuator_armed", required=False)
        if data is None:
            raise ValueError(f"actuator_armed is absent in {path.name}")
        intervals = state_intervals(
            relative_time_s(core, data),
            data["armed"],
            1,
            label="armed",
            series_end_s=float(core.last_timestamp - core.start_timestamp) / 1e6,
        )
    else:
        data = get_topic(core, "vehicle_status", required=False)
        if data is None or "nav_state" not in data:
            raise ValueError(f"vehicle_status.nav_state is absent in {path.name}")
        intervals = state_intervals(
            relative_time_s(core, data),
            data["nav_state"],
            3,
            label="mission",
            series_end_s=float(core.last_timestamp - core.start_timestamp) / 1e6,
        )
    if not intervals:
        raise ValueError(f"no {mode} interval in {path.name}")
    return min(item.start_s for item in intervals), max(item.end_s for item in intervals)


def choose_window(path: Path, args: argparse.Namespace) -> tuple[float, float] | None:
    modes = [name for name in ("airborne", "armed", "mission") if getattr(args, name, False)]
    if len(modes) > 1:
        raise ValueError("choose only one of --airborne, --armed, or --mission")
    if modes and (args.start is not None or args.end is not None):
        raise ValueError("do not combine a named window with --start/--end")
    if modes:
        return signal_window(path, modes[0])
    if args.start is None and args.end is None:
        return None
    return args.start, args.end


def analyze(
    path: Path,
    signals: list[str],
    expressions: list[str],
    aliases: dict[str, str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[str, ...], dict[str, Any]]:
    check_file_size(path, args.max_file_mib)
    specs, parsed = source_specs(signals, expressions, aliases, args.interpolation)
    window = choose_window(path, args)
    start_s, end_s = window if window else (None, None)
    topics = {spec.topic for spec in specs}
    if any(spec.instance == PRIMARY_INSTANCE for spec in specs):
        topics.add("sensors_status_imu")
    log = load_topics(path, topics)
    specs, resolved_instances = resolve_primary_specs(log, specs, start_s=start_s, end_s=end_s)
    ensure_unambiguous_instances(log, specs)
    aligned = align_signals(log, specs, start_s=start_s, end_s=end_s, hz=args.rate, on_missing="skip")
    source_values = {
        spec.label: aligned.values[spec.label]
        for spec in specs
        if spec.label in aligned.values
    }
    values = {signal: source_values[signal] for signal in signals if signal in source_values}
    missing = list(aligned.missing)
    for expression in parsed:
        unavailable = [reference for reference in expression.references if reference not in source_values]
        if unavailable:
            missing.append(expression.label)
            continue
        names = {reference: f"__signal_{index}" for index, reference in enumerate(expression.references)}
        env = {names[reference]: source_values[reference] for reference in expression.references}
        result = np.asarray(eval_expression(expression.tree, env), dtype=np.float64)
        if result.ndim == 0:
            result = np.full(len(aligned.time_s), float(result))
        if result.ndim != 1 or len(result) != len(aligned.time_s):
            raise ValueError(f"expression {expression.label!r} did not produce one value per sample")
        values[expression.label] = result
    metadata = {
        "file": path.name,
        "window_s": [float(aligned.time_s[0]), float(aligned.time_s[-1])],
        "requested_window_s": None if window is None else [
            None if window[0] is None else float(window[0]),
            None if window[1] is None else float(window[1]),
        ],
        "rate_hz": args.rate,
        "missing": missing,
        "sampling": aligned.sampling,
        "linear_interpolation_gap_policy": "do not bridge source gaps longer than five median source periods",
        "source_signals": signals,
        "expressions": [f"{item.label}={item.source}" for item in parsed],
    }
    if resolved_instances:
        metadata["resolved_instances"] = resolved_instances
    return aligned.time_s, values, tuple(missing), metadata


def finite_stats(values: np.ndarray, time_s: np.ndarray | None = None, trend: bool = False) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(array)
    result: dict[str, Any] = {"samples": int(mask.sum())}
    if not np.any(mask):
        return result
    finite = array[mask]
    result.update({
        "mean": float(np.mean(finite)),
        "mean_abs": float(np.mean(np.abs(finite))),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
    })
    if trend and time_s is not None:
        time = np.asarray(time_s, dtype=np.float64)[mask]
        if len(time) >= 2:
            slope = np.polyfit(time - time[0], finite, 1)[0]
            result["slope_per_min"] = float(slope * 60)
            result["first"] = float(finite[0])
            result["last"] = float(finite[-1])
            result["last_minus_first"] = float(finite[-1] - finite[0])
    return result


def longest_finite_run(values: np.ndarray) -> tuple[int, int]:
    mask = np.isfinite(np.asarray(values, dtype=np.float64))
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if not len(starts):
        raise ValueError("signal has no finite samples")
    lengths = ends - starts
    index = int(np.argmax(lengths))
    return int(starts[index]), int(ends[index])


def print_or_write(payload: Any, output: str | None) -> None:
    if output:
        Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def compact_metadata(metadata: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
    if verbose:
        return metadata
    return {key: value for key, value in metadata.items() if key != "sampling"}


def compact_mission(mission: dict[str, Any] | None) -> dict[str, Any] | None:
    if mission is None:
        return None
    return {key: value for key, value in mission.items() if key not in {"item_events", "item_events_truncated"}}


def require_recipe_signals(
    recipe: str,
    values: dict[str, np.ndarray],
    required: Iterable[str],
    missing: Iterable[str],
) -> None:
    """Turn a missing recipe input into an actionable error."""

    unavailable = [name for name in required if name not in values]
    unavailable.extend(name for name in missing if name not in unavailable)
    if unavailable:
        joined = ", ".join(unavailable)
        raise ValueError(f"recipe {recipe!r} requires unavailable signals in the selected window: {joined}")


def common_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--full", action="store_true", help="use the complete common signal window (default)")
    parser.add_argument("--airborne", action="store_true")
    parser.add_argument("--armed", action="store_true")
    parser.add_argument("--mission", action="store_true")
    parser.add_argument("--interpolation", choices=("linear", "previous"), default="linear")
    parser.add_argument("--alias-file", type=Path, default=DEFAULT_ALIAS_FILE)
    parser.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    parser.add_argument("--verbose", action="store_true", help="include per-signal sampling diagnostics")


def add_signal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-s", "--signal", dest="signals", action="append", default=[])
    parser.add_argument("-e", "--expression", dest="expressions", action="append", default=[])


IDENTIFIER_FIELDS = ("device_id", "accel_device_id", "gyro_device_id", "mag_device_id")


def identifier_summary(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in IDENTIFIER_FIELDS:
        if field not in data:
            continue
        values = np.asarray(data[field])
        if np.issubdtype(values.dtype, np.number):
            values = values[np.isfinite(values)]
        unique = np.unique(values)
        converted = [value.item() if isinstance(value, np.generic) else value for value in unique]
        result[field] = converted[0] if len(converted) == 1 else converted
    return result


def primary_identifier_values(
    log: Any,
    field: str,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[int]:
    """Return every non-zero primary ID effective in the selected window."""

    status = get_topic(log, "sensors_status_imu", required=False)
    if status is None or field not in status:
        return []
    values = np.asarray(status[field], dtype=np.float64)
    include = np.isfinite(values) & (values != 0)
    if start_s is not None or end_s is not None:
        time_s = relative_time_s(log, status)
        include &= np.isfinite(time_s)
        if end_s is not None:
            include &= time_s <= end_s
        if start_s is not None:
            before = np.flatnonzero(include & (time_s <= start_s))
            include &= time_s > start_s
            if len(before):
                include[before[-1]] = True
    return sorted({int(value) for value in values[include]})


def _primary_device_ids(
    log: Any,
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[int | None, int | None]:
    accel_ids = primary_identifier_values(log, "accel_device_id_primary", start_s, end_s)
    gyro_ids = primary_identifier_values(log, "gyro_device_id_primary", start_s, end_s)
    changed = {
        name: identifiers
        for name, identifiers in (("accel", accel_ids), ("gyro", gyro_ids))
        if len(identifiers) > 1
    }
    if changed:
        raise ValueError(
            "primary IMU changed within the selected window; split the window at failover "
            f"or use an explicit @INSTANCE (observed IDs: {changed})"
        )
    return (
        accel_ids[0] if accel_ids else None,
        gyro_ids[0] if gyro_ids else None,
    )


def _primary_instance_for_field(
    log: Any,
    field: str,
    start_s: float | None = None,
    end_s: float | None = None,
) -> int:
    """Resolve vehicle_imu_status@primary using sensors_status_imu device IDs."""

    primary_accel, primary_gyro = _primary_device_ids(log, start_s, end_s)
    if primary_accel is None and primary_gyro is None:
        raise ValueError(
            "cannot resolve @primary: sensors_status_imu has no primary device IDs; "
            "use an explicit @INSTANCE"
        )

    if field.startswith("accel"):
        wanted = {primary_accel} if primary_accel is not None else set()
        id_field = "accel_device_id"
    elif field.startswith("gyro"):
        wanted = {primary_gyro} if primary_gyro is not None else set()
        id_field = "gyro_device_id"
    else:
        wanted = {value for value in (primary_accel, primary_gyro) if value is not None}
        id_field = None

    candidates: list[int] = []
    for instance in topic_instances(log, "vehicle_imu_status"):
        data = get_topic(log, "vehicle_imu_status", instance)
        identifiers = identifier_summary(data)
        if id_field is not None:
            value = identifiers.get(id_field)
            values = {value} if isinstance(value, int) else set(value or [])
        else:
            values = {
                value
                for name in ("accel_device_id", "gyro_device_id")
                for value in ((identifiers.get(name),) if isinstance(identifiers.get(name), int) else (identifiers.get(name) or []))
            }
        if values & wanted:
            candidates.append(instance)

    if len(candidates) != 1:
        raise ValueError(
            f"cannot resolve vehicle_imu_status@primary.{field}: "
            f"matched instances {candidates}; use an explicit @INSTANCE"
        )
    return candidates[0]


def resolve_primary_specs(
    log: Any,
    specs: list[SignalSpec],
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[list[SignalSpec], dict[str, int]]:
    resolved: list[SignalSpec] = []
    mapping: dict[str, int] = {}
    for spec in specs:
        if spec.instance != PRIMARY_INSTANCE:
            resolved.append(spec)
            continue
        if spec.topic != "vehicle_imu_status":
            raise ValueError(f"@primary is currently supported only for vehicle_imu_status, got {spec.label!r}")
        instance = _primary_instance_for_field(log, spec.field, start_s, end_s)
        resolved.append(replace(spec, instance=instance))
        mapping[spec.label] = instance
    return resolved, mapping


def command_params_diff(args: argparse.Namespace) -> None:
    if len(args.logs) < 2:
        raise ValueError("params-diff requires at least two logs")
    labels = args.labels or [path.name for path in args.logs]
    if len(labels) != len(args.logs):
        raise ValueError("repeat --label exactly once per log, or omit all labels")
    if len(set(labels)) != len(labels):
        raise ValueError("log labels must be unique; use repeated --label values")

    snapshots = {
        label: dict(read_header(path).initial_parameters)
        for label, path in zip(labels, args.logs)
    }
    counts = {label: len(snapshot) for label, snapshot in snapshots.items()}
    comparison_args = {
        "labels": labels,
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
    }
    all_differences = parameter_diff(args.logs, **comparison_args)
    selected_differences = parameter_diff(args.logs, pattern=args.pattern, **comparison_args)
    selected_parameters = (
        parameter_diff(args.logs, pattern=args.pattern, include_equal=True, **comparison_args)
        if args.include_equal
        else selected_differences
    )
    payload = {
        "logs": {label: str(path) for label, path in zip(labels, args.logs)},
        "parameter_counts": counts,
        "pattern": args.pattern,
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "all_parameter_difference_count": len(all_differences),
        "selected_difference_count": len(selected_differences),
        "selected_parameter_count": len(selected_parameters),
        "include_equal": args.include_equal,
        "selected_missing_parameters": {
            label: [name for name in selected_parameters if name not in snapshot]
            for label, snapshot in snapshots.items()
        },
        "parameters": selected_parameters,
    }
    print_or_write(payload, args.output)


def command_inspect(args: argparse.Namespace) -> None:
    check_file_size(args.log, args.max_file_mib)
    record = build_record(args.log, "none", 1, 1)
    header = read_header(args.log)
    parameters = header.initial_parameters
    selected_parameters = {name: parameters[name] for name in args.parameters if name in parameters}
    missing_parameters = [name for name in args.parameters if name not in parameters]
    payload: dict[str, Any] = {
        "file": args.log.name,
        "classification": record["classification"],
        "quality": record["quality"],
        "duration_s": record["loaded_span_s"],
        "firmware": record["firmware"],
        "hardware": record["hardware"],
        "flight_window_s": record["flight_window_s"],
        "timeline": record["timeline"],
        "mission": compact_mission(record["mission"]),
        "dropouts": {key: record["dropouts"][key] for key in ("count", "total_ms", "max_ms")},
        "message_count": record["messages"]["total_records"],
        "parameter_count": len(parameters),
        "missing_core_topics": record["missing_core_topics"],
    }
    if args.parameters:
        payload["parameters"] = selected_parameters
        payload["missing_parameters"] = missing_parameters
    if args.all_parameters:
        payload["parameters"] = parameters
    if args.details:
        payload["dropout_events"] = record["dropouts"]
        payload["messages"] = record["messages"]
        payload["mission"] = record["mission"]
    print_or_write(payload, args.output)


def command_topics(args: argparse.Namespace) -> None:
    schemas = available_schemas(args.log)
    payload = {name: {"field_count": len(fields), "fields": fields} for name, fields in schemas.items()}
    print_or_write(payload, args.output)


def command_fields(args: argparse.Namespace) -> None:
    check_file_size(args.log, args.max_file_mib)
    schemas = available_schemas(args.log)
    if args.topic not in schemas:
        raise KeyError(f"topic {args.topic!r} is absent; use topics first")
    log = load_topics(args.log, {args.topic})
    instances = []
    for instance in topic_instances(log, args.topic):
        data = get_topic(log, args.topic, instance)
        entry = {
            "instance": instance,
            "fields": schemas[args.topic],
            "sampling": sampling_info(relative_time_s(log, data)),
        }
        identifiers = identifier_summary(data)
        if identifiers:
            entry["identifiers"] = identifiers
        instances.append(entry)
    print_or_write({"topic": args.topic, "fields": schemas[args.topic], "instances": instances}, args.output)


def command_extract(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    time_s, values, missing, metadata = analyze(args.log, args.signals, args.expressions, aliases, args)
    if args.output and args.output.lower().endswith(".csv"):
        with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", *values])
            writer.writerows(zip(time_s, *(values[name] for name in values)))
        print(args.output)
        return
    payload = {"metadata": metadata, "missing": list(missing), "time_s": time_s.tolist(), "signals": {k: v.tolist() for k, v in values.items()}}
    print_or_write(payload, args.output)


def command_stats(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    time_s, values, missing, metadata = analyze(args.log, args.signals, args.expressions, aliases, args)
    payload = {
        "metadata": compact_metadata(metadata, args.verbose),
        "missing": list(missing),
        "stats": {name: finite_stats(value, time_s, args.trend) for name, value in values.items()},
    }
    print_or_write(payload, args.output)


def command_plot(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    time_s, values, missing, metadata = analyze(args.log, args.signals, args.expressions, aliases, args)
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("plot requires matplotlib") from error
    output = args.output or f"{args.log.stem}_signals.png"
    if args.overlay:
        figure, axis = plt.subplots(figsize=(11, 5))
        for name, value in values.items():
            axis.plot(time_s, value, label=name)
        axis.set_xlabel("time since log start (s)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    else:
        height = min(18, max(3, 2.5 * len(values)))
        figure, axes = plt.subplots(len(values), 1, figsize=(11, height), sharex=True, squeeze=False)
        for axis, (name, value) in zip(axes[:, 0], values.items()):
            axis.plot(time_s, value, label=name)
            axis.set_ylabel(name)
            axis.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("time since log start (s)")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print(json.dumps({"output": output, "metadata": compact_metadata(metadata, args.verbose), "missing": list(missing)}, ensure_ascii=False, indent=2))


def command_compare(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    result = {}
    for path in args.logs:
        time_s, values, missing, metadata = analyze(path, args.signals, args.expressions, aliases, args)
        result[path.name] = {
            "metadata": compact_metadata(metadata, args.verbose),
            "missing": list(missing),
            "stats": {name: finite_stats(value, time_s, args.trend) for name, value in values.items()},
        }
    print_or_write(result, args.output)


def command_compare_signals(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    local = argparse.Namespace(**vars(args))
    local.signals = [args.cause, args.effect]
    local.expressions = []
    time_s, values, missing, metadata = analyze(args.log, local.signals, [], aliases, local)
    require_recipe_signals("compare-signals", values, local.signals, missing)
    cause, effect = values[args.cause], values[args.effect]
    finite = np.isfinite(cause) & np.isfinite(effect)
    if finite.sum() < 4:
        raise ValueError("not enough paired finite samples for comparison")
    start, end = longest_finite_run(np.where(finite, 0.0, np.nan))
    if end - start < 4:
        raise ValueError("not enough consecutive paired finite samples for comparison")
    x, y = cause[start:end], effect[start:end]
    x -= np.mean(x)
    y -= np.mean(y)
    max_lag = max(0, int(round(args.max_lag * args.rate)))
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = x[-lag:], y[:lag]
        elif lag > 0:
            left, right = x[:-lag], y[lag:]
        else:
            left, right = x, y
        corr = float(np.corrcoef(left, right)[0, 1]) if len(left) >= 3 and np.std(left) and np.std(right) else float("nan")
        rows.append({"lag_s": lag / args.rate, "correlation": corr, "samples": len(left)})
    valid_rows = [row for row in rows if np.isfinite(row["correlation"])]
    best = max(valid_rows, key=lambda row: abs(row["correlation"])) if valid_rows else None
    payload: dict[str, Any] = {
        "metadata": compact_metadata(metadata, args.verbose),
        "missing": list(missing),
        "cause": args.cause,
        "effect": args.effect,
        "max_lag_s": args.max_lag,
        "effective_window_s": [float(time_s[start]), float(time_s[end - 1])],
        "samples_used": int(end - start),
        "paired_samples_available": int(finite.sum()),
        "paired_samples_omitted": int(finite.sum() - (end - start)),
        "gap_warning": (
            "Only the longest continuous paired-sample run was used; do not treat this lag as representative of omitted segments."
            if finite.sum() > end - start else None
        ),
        "lag_convention": "positive lag means the effect occurs after the cause",
        "best": best,
    }
    if args.all_lags:
        payload["lags"] = rows
    print_or_write(payload, args.output)


def transfer_sampling_safety(path: Path, metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Reject transfer estimates that would get their bandwidth from interpolation."""

    source_rates: dict[str, float] = {}
    unavailable: list[str] = []
    for label, info in metadata.get("sampling", {}).items():
        rate = info.get("median_hz")
        if rate is None or not np.isfinite(rate) or rate <= 0:
            unavailable.append(label)
        else:
            source_rates[label] = float(rate)
    if unavailable:
        raise ValueError(
            f"cannot estimate transfer for {path.name}: source sampling rate is unavailable for {unavailable}"
        )
    if not source_rates:
        raise ValueError(f"cannot estimate transfer for {path.name}: no source sampling metadata")

    slowest_source_hz = min(source_rates.values())
    upsampling = args.rate > slowest_source_hz
    if upsampling and not getattr(args, "allow_upsampling", False):
        raise ValueError(
            f"unsafe transfer request for {path.name}: --rate {args.rate:g} Hz would upsample "
            f"the slowest source ({slowest_source_hz:.3g} Hz). Choose --rate <= "
            f"{slowest_source_hz:.3g} or explicitly pass --allow-upsampling."
        )

    safe_max_hz = min(args.rate / 2, slowest_source_hz / 2)
    unsafe_frequencies = [frequency for frequency in args.at_hz if frequency >= safe_max_hz]
    if unsafe_frequencies:
        raise ValueError(
            f"unsafe transfer request for {path.name}: --at-hz must be below {safe_max_hz:.3g} Hz "
            f"(the lower of resampled and raw-source Nyquist); got {unsafe_frequencies}"
        )
    return {
        "source_median_hz": source_rates,
        "requested_grid_hz": args.rate,
        "safe_frequency_range_hz": [0.0, safe_max_hz],
        "upsampling": upsampling,
        "upsampling_warning": (
            "The grid is denser than the slowest source; no new bandwidth is created. "
            "Interpret confidence and phase conservatively."
            if upsampling else None
        ),
    }


def command_transfer(args: argparse.Namespace) -> None:
    """Estimate gain, phase, and coherence for a cause/effect pair."""

    aliases = load_aliases(args.alias_file)
    if not args.at_hz:
        raise ValueError("provide at least one --at-hz frequency to report gain, phase, and coherence")
    if args.min_segments <= 0:
        raise ValueError("--min-segments must be positive")
    if args.nperseg is not None and args.nperseg <= 0:
        raise ValueError("--nperseg must be positive")
    invalid_frequencies = [
        frequency for frequency in args.at_hz
        if not np.isfinite(frequency) or frequency < 0
    ]
    if invalid_frequencies:
        raise ValueError(
            f"--at-hz values must be finite and non-negative: {invalid_frequencies}"
        )
    local = argparse.Namespace(**vars(args))
    local.signals = [args.cause, args.effect]
    local.expressions = []
    time_s, values, missing, metadata = analyze(args.log, local.signals, [], aliases, local)
    require_recipe_signals("transfer", values, local.signals, missing)
    sampling_safety = transfer_sampling_safety(args.log, metadata, args)
    cause, effect = values[args.cause], values[args.effect]
    finite = np.isfinite(cause) & np.isfinite(effect)
    start, end = longest_finite_run(np.where(finite, 0.0, np.nan))
    result = transfer_function(
        cause[start:end],
        effect[start:end],
        args.rate,
        nperseg=args.nperseg,
        min_segments=args.min_segments,
    )
    points = [at_frequency(result, frequency) for frequency in args.at_hz]
    payload = {
        "metadata": compact_metadata(metadata, args.verbose),
        "missing": list(missing),
        "cause": args.cause,
        "effect": args.effect,
        "effective_window_s": [float(time_s[start]), float(time_s[end - 1])],
        "samples_used": int(end - start),
        "segments": result.segments,
        "frequency_resolution_hz": result.frequency_resolution_hz,
        "sampling_safety": sampling_safety,
        "at_frequency": points,
        "interpretation_warning": "Gain, phase, and coherence describe signal compatibility; they do not establish causality by themselves.",
    }
    print_or_write(payload, args.output)


def validate_spectrum_sampling(path: Path, metadata: dict[str, Any], args: argparse.Namespace) -> None:
    problems: list[str] = []
    for label, info in metadata.get("sampling", {}).items():
        source_hz = info.get("median_hz")
        if source_hz is None or not np.isfinite(source_hz):
            problems.append(f"{label}: source sampling rate is unavailable")
            continue
        source_hz = float(source_hz)
        if args.high_hz >= source_hz / 2:
            problems.append(
                f"{label}: requested {args.high_hz:g} Hz reaches/exceeds source Nyquist "
                f"{source_hz / 2:.3g} Hz (median source rate {source_hz:.3g} Hz)"
            )
        if args.rate > source_hz * 1.05:
            problems.append(
                f"{label}: requested resample rate {args.rate:g} Hz would upsample "
                f"median source rate {source_hz:.3g} Hz"
            )
    if problems:
        raise ValueError(f"unsafe spectrum request for {path.name}: " + "; ".join(problems))


def command_spectrum(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    payload: dict[str, Any] = {}
    for path in args.logs:
        time_s, values, missing, metadata = analyze(path, args.signals, args.expressions, aliases, args)
        validate_spectrum_sampling(path, metadata, args)
        summaries: dict[str, Any] = {}
        for name, value in values.items():
            array = np.asarray(value, dtype=np.float64)
            start, end = longest_finite_run(array)
            valid = array[start:end]
            spectrum = welch_peak(
                valid,
                args.rate,
                args.low_hz,
                args.high_hz,
                nperseg=args.nperseg,
                min_segments=args.min_segments,
            )
            summaries[name] = {
                "band_hz": [args.low_hz, args.high_hz],
                "band_rms": band_rms(valid, args.rate, args.low_hz, args.high_hz),
                "peak_frequency_hz": spectrum.peak_frequency_hz,
                "peak_psd": spectrum.peak_psd,
                "effective_window_s": [float(time_s[start]), float(time_s[end - 1])],
                "samples_used": len(valid),
                "samples_omitted": len(array) - len(valid),
                "segments": spectrum.segments,
                "frequency_resolution_hz": spectrum.frequency_resolution_hz,
            }
        payload[path.name] = {
            "metadata": compact_metadata(metadata, args.verbose),
            "missing": list(missing),
            "spectrum": summaries,
        }
    print_or_write(payload, args.output)


def command_health(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {}
    for path in args.logs:
        check_file_size(path, args.max_file_mib)
        log = load_topics(path, {"cpuload"})
        quality, dropouts = quality_summary(log)
        cpu = get_topic(log, "cpuload", required=False)
        resources: dict[str, Any] = {}
        if cpu is not None:
            time_s = relative_time_s(log, cpu)
            for field in ("load", "ram_usage"):
                if field in cpu:
                    resources[field] = finite_stats(np.asarray(cpu[field]), time_s, trend=True)

        stack_tasks: dict[str, dict[str, Any]] = {}
        for item in log.logged_messages:
            text = item.message.rstrip()
            match = LOW_STACK_RE.search(text)
            if not match:
                continue
            task = match.group("task")
            bytes_left = int(match.group("bytes"))
            time_s = float(item.timestamp - log.start_timestamp) / 1e6
            summary = stack_tasks.setdefault(
                task,
                {"count": 0, "min_bytes_left": bytes_left, "first_time_s": time_s, "last_time_s": time_s},
            )
            summary["count"] += 1
            summary["min_bytes_left"] = min(summary["min_bytes_left"], bytes_left)
            summary["last_time_s"] = time_s

        payload[path.name] = {
            "duration_s": float(log.last_timestamp - log.start_timestamp) / 1e6,
            "quality": quality,
            "cpu_and_memory": resources,
            "dropouts": {key: dropouts[key] for key in ("count", "total_ms", "max_ms")},
            "low_stack": stack_tasks,
        }
    print_or_write(payload, args.output)


def cropped_topic_time(
    log: Any,
    data: dict[str, Any],
    start_s: float | None,
    end_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    time_s = relative_time_s(log, data)
    mask = np.isfinite(time_s)
    if start_s is not None:
        mask &= time_s >= start_s
    if end_s is not None:
        mask &= time_s <= end_s
    return time_s[mask], mask


def finite_max(values: Any) -> int | float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return None
    value = float(np.max(finite))
    return int(value) if value.is_integer() else value


def command_vibration_recipe(args: argparse.Namespace) -> None:
    check_file_size(args.log, args.max_file_mib)
    requested_window = choose_window(args.log, args)
    start_s, end_s = requested_window if requested_window else (None, None)
    topics = {"vehicle_imu_status", "sensors_status_imu", "sensor_accel", "sensor_gyro"}
    log = load_topics(args.log, topics)

    sensors_status = get_topic(log, "sensors_status_imu", required=False)
    primary_accel_ids = primary_identifier_values(log, "accel_device_id_primary", start_s, end_s)
    primary_gyro_ids = primary_identifier_values(log, "gyro_device_id_primary", start_s, end_s)
    primary_accel_id = primary_accel_ids[0] if len(primary_accel_ids) == 1 else None
    primary_gyro_id = primary_gyro_ids[0] if len(primary_gyro_ids) == 1 else None
    missing: list[str] = []
    if sensors_status is None:
        missing.append("sensors_status_imu")

    instance_rows: list[dict[str, Any]] = []
    effective_starts: list[float] = []
    effective_ends: list[float] = []
    for instance in topic_instances(log, "vehicle_imu_status"):
        data = get_topic(log, "vehicle_imu_status", instance)
        time_s, mask = cropped_topic_time(log, data, start_s, end_s)
        if not len(time_s):
            continue
        effective_starts.append(float(time_s[0]))
        effective_ends.append(float(time_s[-1]))
        identifiers = identifier_summary(data)
        accel_id = identifiers.get("accel_device_id")
        gyro_id = identifiers.get("gyro_device_id")
        stats_payload = {
            field: finite_stats(np.asarray(data[field])[mask], time_s, trend=True)
            for field in ("accel_vibration_metric", "gyro_vibration_metric", "delta_angle_coning_metric")
            if field in data
        }
        clipping = {
            field: finite_max(np.asarray(data[field])[mask])
            for field in (
                "accel_clipping[0]", "accel_clipping[1]", "accel_clipping[2]",
                "gyro_clipping[0]", "gyro_clipping[1]", "gyro_clipping[2]",
            )
            if field in data
        }
        errors = {
            field: finite_max(np.asarray(data[field])[mask])
            for field in ("accel_error_count", "gyro_error_count")
            if field in data
        }
        instance_rows.append({
            "instance": instance,
            "identifiers": identifiers,
            "primary_accel": isinstance(accel_id, int) and accel_id in primary_accel_ids,
            "primary_gyro": isinstance(gyro_id, int) and gyro_id in primary_gyro_ids,
            "sampling": sampling_info(time_s),
            "stats": stats_payload,
            "clipping_max": clipping,
            "error_count_max": errors,
        })
    if not instance_rows:
        raise ValueError(f"vehicle_imu_status is absent or empty in {args.log.name}")

    raw_sampling: dict[str, list[dict[str, Any]]] = {}
    for topic in ("sensor_accel", "sensor_gyro"):
        rows: list[dict[str, Any]] = []
        for instance in topic_instances(log, topic):
            data = get_topic(log, topic, instance)
            time_s, _mask = cropped_topic_time(log, data, start_s, end_s)
            if not len(time_s):
                continue
            rows.append({
                "instance": instance,
                "identifiers": identifier_summary(data),
                "sampling": sampling_info(time_s),
            })
        raw_sampling[topic] = rows
        if not rows:
            missing.append(topic)

    primary_instances = [
        row for row in instance_rows
        if row["primary_accel"] or row["primary_gyro"]
    ]
    warnings = [
        "Use sensor_combined or another high-rate topic for vibration spectra; sensor_accel/sensor_gyro may be logged at 1 Hz.",
        "Compare primary_instances before comparing IMU vibration metrics across logs.",
    ]
    if not primary_accel_ids and not primary_gyro_ids:
        warnings.append("No primary IMU IDs were logged in this window; use an explicit instance for quantitative comparison.")
    if len(primary_accel_ids) > 1 or len(primary_gyro_ids) > 1:
        warnings.append("Primary IMU changed within this window; primary_instances contains every observed primary, not one static primary.")

    payload = {
        "recipe": "vibration",
        "metadata": {
            "file": args.log.name,
            "window_s": [min(effective_starts), max(effective_ends)],
            "requested_window_s": None if requested_window is None else list(requested_window),
            "sampling_mode": "native topic samples",
            "metric_semantics": {
                "accel_vibration_metric": "EWMA of adjacent integrated-acceleration differences; not a frequency or amplitude",
                "gyro_vibration_metric": "EWMA of adjacent integrated-angle differences; not a frequency or amplitude",
                "delta_angle_coning_metric": "coning correction magnitude; inspect trends and clipping before attributing a bias",
            },
            "primary_ids": {
                "accel_device_id": primary_accel_id,
                "gyro_device_id": primary_gyro_id,
            },
            "observed_primary_ids": {
                "accel_device_id": primary_accel_ids,
                "gyro_device_id": primary_gyro_ids,
            },
            "primary_selection": "stable" if primary_accel_id is not None or primary_gyro_id is not None else "unavailable-or-changed",
        },
        "missing": missing,
        "primary_instances": primary_instances,
        "instances": instance_rows,
        "raw_sampling": raw_sampling,
        "warnings": warnings,
    }
    print_or_write(payload, args.output)


def command_attitude_tracking_recipe(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    actual_names = [f"vehicle_attitude.q[{index}]" for index in range(4)]
    desired_names = [f"vehicle_attitude_setpoint.q_d[{index}]" for index in range(4)]
    rate_actual = [f"vehicle_angular_velocity.xyz[{index}]" for index in range(3)]
    rate_desired = ["vehicle_rates_setpoint.roll", "vehicle_rates_setpoint.pitch", "vehicle_rates_setpoint.yaw"]
    signals = [*actual_names, *desired_names, *rate_actual, *rate_desired]
    time_s, values, missing, metadata = analyze(args.log, signals, [], aliases, args)
    require_recipe_signals("attitude-tracking", values, signals, missing)

    actual = np.column_stack([values[name] for name in actual_names])
    desired = np.column_stack([values[name] for name in desired_names])
    finite = np.all(np.isfinite(actual), axis=1) & np.all(np.isfinite(desired), axis=1)
    actual_norm = np.linalg.norm(actual[finite], axis=1)
    desired_norm = np.linalg.norm(desired[finite], axis=1)
    valid_norm = (actual_norm > 0) & (desired_norm > 0)
    dots = np.sum(
        actual[finite][valid_norm] / actual_norm[valid_norm, None]
        * desired[finite][valid_norm] / desired_norm[valid_norm, None],
        axis=1,
    )
    attitude_error_deg = np.degrees(2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))

    rate_error_names = ("roll_rate_error", "pitch_rate_error", "yaw_rate_error")
    rate_errors = {
        label: values[desired_name] - values[actual_name]
        for label, desired_name, actual_name in zip(rate_error_names, rate_desired, rate_actual)
    }
    payload = {
        "recipe": "attitude-tracking",
        "metadata": {
            **compact_metadata(metadata, args.verbose),
            "quaternion_order": "w, x, y, z",
            "attitude_error_definition": "shortest rotation angle between vehicle_attitude.q and vehicle_attitude_setpoint.q_d",
            "quaternion_interpolation": args.interpolation,
        },
        "missing": list(missing),
        "stats": {
            "attitude_error_deg": finite_stats(attitude_error_deg),
            **{name: finite_stats(error, time_s, trend=True) for name, error in rate_errors.items()},
        },
    }
    print_or_write(payload, args.output)


def command_horizontal_control_recipe(args: argparse.Namespace) -> None:
    aliases = load_aliases(args.alias_file)
    header = read_header(args.log)
    parameters = header.initial_parameters
    expressions = [
        "local_position_setpoint_error_n=vehicle_local_position_setpoint.x-vehicle_local_position.x",
        "local_position_setpoint_error_e=vehicle_local_position_setpoint.y-vehicle_local_position.y",
        "trajectory_position_error_n=trajectory_setpoint.position[0]-vehicle_local_position.x",
        "trajectory_position_error_e=trajectory_setpoint.position[1]-vehicle_local_position.y",
        "velocity_error_n=vehicle_local_position_setpoint.vx-vehicle_local_position.vx",
        "velocity_error_e=vehicle_local_position_setpoint.vy-vehicle_local_position.vy",
        "internal_velocity_minus_trajectory_velocity_n=vehicle_local_position_setpoint.vx-trajectory_setpoint.velocity[0]",
        "internal_velocity_minus_trajectory_velocity_e=vehicle_local_position_setpoint.vy-trajectory_setpoint.velocity[1]",
    ]
    signals = [
        "trajectory_setpoint.position[0]", "trajectory_setpoint.position[1]",
        "trajectory_setpoint.velocity[0]", "trajectory_setpoint.velocity[1]",
        "vehicle_local_position_setpoint.acceleration[0]", "vehicle_local_position_setpoint.acceleration[1]",
        "vehicle_local_position.ax", "vehicle_local_position.ay",
        "vehicle_attitude_setpoint.roll_body", "vehicle_attitude_setpoint.pitch_body",
    ]
    time_s, values, missing, metadata = analyze(args.log, signals, expressions, aliases, args)
    require_recipe_signals("horizontal-control", values, [*signals, *[item.split("=", 1)[0] for item in expressions]], missing)
    gain_names = (
        "MPC_XY_P", "MPC_XY_VEL_P_ACC", "MPC_XY_VEL_I_ACC", "MPC_XY_VEL_D_ACC",
        "MPC_XY_ERR_MAX", "MPC_ACC_HOR", "MPC_ACC_HOR_MAX",
    )
    requested_window = metadata.get("requested_window_s")
    selected_start, selected_end = (
        metadata["window_s"]
        if requested_window is None or any(bound is None for bound in requested_window)
        else requested_window
    )
    effective_parameters = {name: parameters.get(name) for name in gain_names}
    parameter_changes = [
        {
            "time_s": float(timestamp - header.start_timestamp) / 1e6,
            "name": name,
            "value": value.item() if isinstance(value, np.generic) else value,
        }
        for timestamp, name, value in sorted(header.changed_parameters, key=lambda item: item[0])
        if name in gain_names and selected_start <= float(timestamp - header.start_timestamp) / 1e6 <= selected_end
    ]
    for timestamp, name, value in sorted(header.changed_parameters, key=lambda item: item[0]):
        changed_at_s = float(timestamp - header.start_timestamp) / 1e6
        if name in effective_parameters and changed_at_s <= selected_start:
            effective_parameters[name] = value.item() if isinstance(value, np.generic) else value

    try:
        window_start_xy_p = float(effective_parameters["MPC_XY_P"])
    except (TypeError, ValueError):
        window_start_xy_p = float("nan")
    if np.isfinite(window_start_xy_p):
        values["window_start_gain_position_feedback_velocity_n"] = (
            window_start_xy_p * values["trajectory_position_error_n"]
        )
        values["window_start_gain_position_feedback_velocity_e"] = (
            window_start_xy_p * values["trajectory_position_error_e"]
        )
        values["velocity_constraint_residual_n"] = (
            values["internal_velocity_minus_trajectory_velocity_n"]
            - values["window_start_gain_position_feedback_velocity_n"]
        )
        values["velocity_constraint_residual_e"] = (
            values["internal_velocity_minus_trajectory_velocity_e"]
            - values["window_start_gain_position_feedback_velocity_e"]
        )
    legacy_aliases = {
        "position_error_n": "local_position_setpoint_error_n",
        "position_error_e": "local_position_setpoint_error_e",
        "position_feedback_velocity_n": "internal_velocity_minus_trajectory_velocity_n",
        "position_feedback_velocity_e": "internal_velocity_minus_trajectory_velocity_e",
    }
    for legacy_name, current_name in legacy_aliases.items():
        values[legacy_name] = values[current_name]
    payload = {
        "recipe": "horizontal-control",
        "metadata": {
            **compact_metadata(metadata, args.verbose),
            "coordinates": "NED: x/vx/acceleration[0] are north; y/vy/acceleration[1] are east",
            "setpoint_semantics": "internal_velocity_minus_trajectory_velocity is the final internal velocity setpoint minus trajectory feed-forward; it can include XY velocity constraints and is not, by itself, pure position feedback",
            "initial_horizontal_parameters": {name: parameters.get(name) for name in gain_names},
            "effective_horizontal_parameters_at_window_start": effective_parameters,
            "parameter_semantics": "Initial values come from the ULog header. Effective-at-window-start values replay logged parameter changes before the selected window; time-varying gains within the window are not applied to the derived feedback signals.",
            "horizontal_parameter_changes_in_window": parameter_changes,
            "deprecated_stats_aliases": legacy_aliases,
        },
        "missing": list(missing),
        "stats": {name: finite_stats(value, time_s, trend=True) for name, value in values.items()},
        "window_warning": "Analyze trajectory transitions and settled holds in separate windows before comparing RMS or inferring controller behavior.",
        "parameter_warning": (
            "Selected horizontal parameters changed inside this window; do not treat one set of RMS statistics as one fixed-gain response."
            if parameter_changes else None
        ),
    }
    print_or_write(payload, args.output)


RECIPES = {
    "vertical": [
        "vehicle_local_position.z",
        "vehicle_local_position.vz",
        "vehicle_local_position_setpoint.z",
        "vehicle_local_position_setpoint.vz",
        "vehicle_gps_position.vel_d_m_s",
        "vehicle_acceleration.xyz[2]",
    ],
    "attitude": ["vehicle_attitude.q[0]", "vehicle_attitude.q[1]", "vehicle_attitude.q[2]", "vehicle_attitude.q[3]", "vehicle_angular_velocity.xyz[0]", "vehicle_angular_velocity.xyz[1]", "vehicle_angular_velocity.xyz[2]"],
    "attitude-tracking": None,
    "horizontal-control": None,
    "vibration": None,
    "gps-ekf": ["vehicle_local_position.vz", "vehicle_gps_position.vel_d_m_s", "estimator_innovations.gps_vvel", "estimator_innovation_test_ratios.gps_vvel"],
    "motors": ["actuator_motors.control[0]", "actuator_motors.control[1]", "actuator_motors.control[2]", "actuator_motors.control[3]"],
}


def command_recipe(args: argparse.Namespace) -> None:
    if args.name not in RECIPES:
        raise ValueError(f"unknown recipe {args.name!r}; choose from {sorted(RECIPES)}")
    if args.name == "vibration":
        command_vibration_recipe(args)
        return
    if args.name == "attitude-tracking":
        command_attitude_tracking_recipe(args)
        return
    if args.name == "horizontal-control":
        command_horizontal_control_recipe(args)
        return
    aliases = load_aliases(args.alias_file)
    local = argparse.Namespace(**vars(args))
    local.signals = RECIPES[args.name]
    local.expressions = []
    time_s, values, missing, metadata = analyze(args.log, local.signals, [], aliases, local)
    payload = {
        "recipe": args.name,
        "metadata": compact_metadata(metadata, args.verbose),
        "missing": list(missing),
        "stats": {name: finite_stats(value, time_s, True) for name, value in values.items()},
    }
    print_or_write(payload, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reusable, selective PX4 ULog inspection and signal analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="compact flight metadata and classification")
    inspect.add_argument("log", type=Path)
    inspect.add_argument("--parameter", dest="parameters", action="append", default=[], metavar="NAME", help="include one initial parameter; repeat as needed")
    inspect.add_argument("--all-parameters", action="store_true", help="include the complete initial parameter table")
    inspect.add_argument("--details", action="store_true", help="include messages, dropout events, and mission item events")
    inspect.add_argument("--output")
    inspect.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    inspect.set_defaults(func=command_inspect)

    params_diff = sub.add_parser("params-diff", help="compare complete header-stored parameter snapshots")
    params_diff.add_argument("logs", type=Path, nargs="+")
    params_diff.add_argument("--pattern", help="regular expression selecting parameters for output")
    params_diff.add_argument("--include-equal", action="store_true", help="include equal parameters matching --pattern")
    params_diff.add_argument("--abs-tol", type=float, default=0.0, help="absolute numeric equality tolerance")
    params_diff.add_argument("--rel-tol", type=float, default=0.0, help="relative numeric equality tolerance")
    params_diff.add_argument("--label", dest="labels", action="append", help="label one log; repeat in log order")
    params_diff.add_argument("--output")
    params_diff.set_defaults(func=command_params_diff)

    topics = sub.add_parser("topics", help="list header-declared topics and fields")
    topics.add_argument("log", type=Path)
    topics.add_argument("--output")
    topics.set_defaults(func=command_topics)

    fields = sub.add_parser("fields", help="show fields and loaded instances for one topic")
    fields.add_argument("log", type=Path)
    fields.add_argument("topic")
    fields.add_argument("--output")
    fields.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    fields.set_defaults(func=command_fields)

    for name, function, help_text in (("extract", command_extract, "extract aligned samples"), ("stats", command_stats, "descriptive statistics"), ("trend", command_stats, "statistics with a time trend"), ("plot", command_plot, "plot selected signals")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("log", type=Path)
        add_signal_args(command)
        common_window_args(command)
        if name == "extract":
            destination = command.add_mutually_exclusive_group(required=True)
            destination.add_argument("--output", help="write CSV or JSON samples to a file")
            destination.add_argument("--stdout", action="store_true", help="print samples to stdout intentionally")
        else:
            command.add_argument("--output")
        if name in {"stats", "trend"}:
            command.add_argument("--trend", action="store_true", default=name == "trend")
        if name == "plot":
            command.add_argument("--overlay", action="store_true", help="overlay signals only when their units are comparable")
        command.set_defaults(func=function)

    compare = sub.add_parser("compare", help="compare signals across logs")
    compare.add_argument("logs", type=Path, nargs="+")
    add_signal_args(compare)
    common_window_args(compare)
    compare.add_argument("--output")
    compare.add_argument("--trend", action="store_true")
    compare.set_defaults(func=command_compare)

    compare_signals = sub.add_parser("compare-signals", help="cross-correlation and best lag for two signals")
    compare_signals.add_argument("log", type=Path)
    compare_signals.add_argument("--cause", required=True)
    compare_signals.add_argument("--effect", required=True)
    compare_signals.add_argument("--max-lag", type=float, default=2.0)
    common_window_args(compare_signals)
    compare_signals.add_argument("--output")
    compare_signals.add_argument("--all-lags", action="store_true", help="include every evaluated lag instead of only the best result")
    compare_signals.set_defaults(func=command_compare_signals)

    transfer = sub.add_parser("transfer", help="estimate gain, phase, and coherence for two aligned signals")
    transfer.add_argument("log", type=Path)
    transfer.add_argument("--cause", required=True)
    transfer.add_argument("--effect", required=True)
    transfer.add_argument("--at-hz", type=float, action="append", default=[], required=True, help="report the nearest frequency; repeat as needed")
    common_window_args(transfer)
    transfer.add_argument("--allow-upsampling", action="store_true", help="allow a grid faster than a source; raw-source Nyquist still limits --at-hz")
    transfer.add_argument("--nperseg", type=int)
    transfer.add_argument("--min-segments", type=int, default=4)
    transfer.add_argument("--output")
    transfer.set_defaults(func=command_transfer)

    spectrum = sub.add_parser("spectrum", help="compare dominant frequency and band RMS across logs")
    spectrum.add_argument("logs", type=Path, nargs="+")
    add_signal_args(spectrum)
    common_window_args(spectrum)
    spectrum.add_argument("--low-hz", type=float, required=True)
    spectrum.add_argument("--high-hz", type=float, required=True)
    spectrum.add_argument("--nperseg", type=int)
    spectrum.add_argument("--min-segments", type=int, default=4)
    spectrum.add_argument("--output")
    spectrum.set_defaults(func=command_spectrum)

    health = sub.add_parser("health", help="summarize CPU, memory, dropouts, and low-stack warnings")
    health.add_argument("logs", type=Path, nargs="+")
    health.add_argument("--max-file-mib", type=float, default=DEFAULT_MAX_FILE_MIB)
    health.add_argument("--output")
    health.set_defaults(func=command_health)

    recipe = sub.add_parser("recipe", help="run a reusable signal group")
    recipe.add_argument("name", choices=sorted(RECIPES))
    recipe.add_argument("log", type=Path)
    common_window_args(recipe)
    recipe.add_argument("--output")
    recipe.set_defaults(func=command_recipe)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "signals"):
        args.signals = []
    if not hasattr(args, "expressions"):
        args.expressions = []
    if not hasattr(args, "trend"):
        args.trend = False
    try:
        args.func(args)
    except Exception as error:
        print(f"ulog-tool: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
