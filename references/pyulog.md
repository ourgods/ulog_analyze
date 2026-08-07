# PX4 ULog access reference

## Cost-aware loading

```python
from uloglib import read_header, scan_log, load_topics, load_for_specs

header = read_header(LOG_PATH)                  # schemas + initial parameters only
scan = scan_log(LOG_PATH)                       # full byte scan, no topic arrays
log = load_topics(LOG_PATH, {"vehicle_status", "vehicle_local_position"})
log = load_for_specs(LOG_PATH, SIGNAL_SPECS)      # derives the explicit union from declarations
```

`pyulog.ULog(path, parse_header_only=True)` stops after definitions. `ULog(path, message_name_filter_list=[])` scans the complete file for parameters, messages, and dropouts without retaining topic data. `ULog(path, message_name_filter_list=[...])` retains only named topics. Bare `ULog(path)` loads all topics.

Header-only mode does not provide flight duration, messages, or dropouts. An empty topic scan does not produce a meaningful `last_timestamp`; calculate spans from selected topic timestamps.

## Schemas and topics

```python
from uloglib import available_schemas, get_topic, topic_instances

schemas = available_schemas(LOG_PATH)
print(schemas["vehicle_local_position"])

data = get_topic(log, "vehicle_local_position", instance=0)
print(list(data))
print(topic_instances(log, "vehicle_imu_status"))
```

Array-like uORB fields use literal keys such as `xyz[2]`, `q[0]`, and `control[5]`.

## Time, windows, and alignment

Topic timestamps are microseconds on the PX4 boot clock. `uloglib` exposes log-relative seconds:

```python
from uloglib import SignalSpec, read_series, align_signals

spec = SignalSpec("vz", "vehicle_local_position", "vz", scale=-1)
raw = read_series(log, spec, start_s=30, end_s=90)

aligned = align_signals(
    log,
    [
        spec,
        SignalSpec("armed", "actuator_armed", "armed", interpolation="previous"),
    ],
    start_s=30,
    end_s=90,
    hz=10,
    on_missing="skip",
)
print(aligned.missing)
```

Use `interpolation="linear"` only for continuous signals. Use `"previous"` for modes, flags, and counters. Alignment returns NaN outside each source signal's range rather than silently extrapolating.

Use `read_series(..., required=False)` or `align_signals(..., on_missing="skip")` when comparing heterogeneous logs where a topic may legitimately be absent. Report the returned missing labels.

Use `change_timeline()` for armed/mode/takeoff transitions and `summarize_series()` for batches of raw-field statistics. Both accept `on_missing="skip"` and return missing labels rather than hiding absent data.

Sparse discrete topics have previous-value semantics: the last sample time is not necessarily the state end. Build intervals with an explicit enclosing end:

```python
from uloglib import relative_time_s, state_intervals

data = get_topic(log, "takeoff_status")
intervals = state_intervals(
    relative_time_s(log, data),
    data["takeoff_state"],
    expected=5,
    label="flight",
    series_end_s=(log.last_timestamp - log.start_timestamp) / 1e6,
)
```

Use `unwrap_finite(angle, period=360)` for degree signals containing NaN gaps. It unwraps each contiguous finite run independently.

## Parameters, messages, and changes

```python
from uloglib import parameter_diff, scan_log

diff = parameter_diff([LOG_A, LOG_B], pattern=r"^(EKF2|MPC)_", labels=["baseline", "test"])

scan = scan_log(LOG_A)
for timestamp_us, name, value in scan.changed_parameters:
    print((timestamp_us - scan.start_timestamp) / 1e6, name, value)
for item in scan.logged_messages:
    print((item.timestamp - scan.start_timestamp) / 1e6, item.log_level, item.message.rstrip())
```

Initial parameters are snapshots stored in that log. An empty `changed_parameters` list does not prove the configuration never changed before logging.

## Common pitfalls

- A log can start while already armed or airborne. Log-relative zero is not necessarily boot, arming, or takeoff.
- PX4 local coordinates are NED: `z` and `vz` are positive down. Apply `scale=-1` only when upward presentation is intended.
- Check instances and field availability before assuming instance `0`.
- Inspect sample times and gaps before resampling or spectral analysis.
- Do not interpolate discrete states as continuous values.
- Do not use the final sample carrying a discrete value as that state's end; use the next change time or an explicit enclosing end.
- A true estimator event flag can be latched; do not count every true sample as a new event.
- An innovation proves measurement-prediction disagreement, not which side is wrong.
- Compare matched flight phases and setpoint conditions, not whole-flight ranges.
- A coherence estimate from one or two segments is not evidence. Use the validated helpers.
