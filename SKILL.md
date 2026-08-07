---
name: inspect-px4-ulog
description: Use the bundled ulog-tool CLI to inspect, extract, plot, compare parameters and signals, calculate statistics, spectra, transfer functions and coherence, resolve primary IMU instances, and check system health from PX4 ULog files. Use when Codex needs PX4 ULog topics, fields, metadata, parameter changes, windows, aligned signals, trends, plots, oscillation frequencies, vibration, resource warnings, or multi-log comparisons. Prefer reusable CLI commands and recipes over one-off parsing scripts.
---

# PX4 ULog Tool

Use `scripts/ulog_tool.py`. It is a data-access and signal-calculation tool, not a root-cause engine. Select the signals and windows required by the user's question; explain causes only from the resulting measurements.

`@primary` is currently supported only for `vehicle_imu_status`, for example `vehicle_imu_status@primary.gyro_vibration_metric`. It resolves the PX4-selected device from `sensors_status_imu` and reports the resolution in metadata. It rejects a window containing IMU failover rather than silently selecting the last primary; split that window or use an explicit `@INSTANCE`. Never compare `vehicle_imu_status@0` across logs without checking `recipe vibration` first.

Do not create a case directory, a ULog parser, or a case-specific analysis script when an existing command can perform the operation.

## Log Database

The sole canonical log-synopsis database is `E:\UAV\hawkeye-0.3.0-windows-x64\document26,7\analysis\flight_index.json`. Query it when a task needs cross-log selection or previous flight context; it is not mandatory for an isolated single-log lookup.

`task_synopsis` must describe what approximately happened in that log: the activity, major mode sequence, test condition, and known invalidating event such as a motor connection fault. Do not replace the activity synopsis with possible future uses or a root-cause conclusion.

When a new log appears or the current analysis confirms or corrects what happened in an existing log, update only the matching record's `task_synopsis` in `flight_index.json`. Preserve recorded classification, metrics, and provenance unless those fields themselves are being corrected from evidence. Never create or maintain a second index, Markdown summary, case directory, or report.

Run `scripts/flight_catalog.py` only when new logs must be added to this database. Name the new files explicitly and set bounded scan limits; do not run cataloging before routine log analysis.

```powershell
python scripts/flight_catalog.py LOG_DIR --output ANALYSIS_DIR --names NEW_LOG.ulg --max-new-files 1 --max-file-mib 64 --max-scan-mib 128
```

Before a command reads samples, state the selected logs, signal expressions with their sign and coordinate meaning, requested window, rate, and any explicit large-file limit. Keep every numeric result tied to its log and effective window; do not combine different windows as one evidence set.

## Inspect Logs

```powershell
python scripts/ulog_tool.py inspect LOG.ulg
python scripts/ulog_tool.py inspect LOG.ulg --parameter MC_ROLL_P --parameter MC_ROLLRATE_D
python scripts/ulog_tool.py inspect LOG.ulg --all-parameters --details
python scripts/ulog_tool.py topics LOG.ulg
python scripts/ulog_tool.py fields LOG.ulg vehicle_local_position
python scripts/ulog_tool.py params-diff BASE.ulg TEST.ulg
python scripts/ulog_tool.py params-diff BASE.ulg TEST.ulg --pattern '^(MC_|MPC_)' --abs-tol 0.0001
```

`inspect` is compact by default. Use `--parameter NAME` for selected initial parameters, `--all-parameters` only when the full table is needed, and `--details` for messages, dropout events, and mission item events.

`topics` reads only the header. Commands that load samples default to a 64 MiB per-file limit. Raise it deliberately for a selected large log:

```powershell
python scripts/ulog_tool.py trend log89.ulg -e "hawkeye_vz=-vehicle_local_position.vz" --max-file-mib 128
```

Selective loading still scans ULog bytes, but it retains only named topics.

Use `params-diff` before claiming that configurations are equal or different. Its output always reports the complete parameter-difference count even when `--pattern` filters the displayed parameters. Use `--abs-tol` for manually rounded values.

## Signals And Expressions

Use `-s TOPIC.FIELD` for a direct signal and `-e NAME=EXPRESSION` for simple arithmetic. Array fields use the ULog spelling such as `xyz[2]`.

When a topic has multiple instances, the CLI rejects an implicit instance. Run `fields` to see instance identifiers, then use `TOPIC@INSTANCE.FIELD`. Do not assume instance 0 is primary.

```powershell
python scripts/ulog_tool.py extract LOG.ulg -s vehicle_local_position.vz -s vehicle_gps_position.vel_d_m_s --output out.csv
python scripts/ulog_tool.py fields LOG.ulg vehicle_imu_status
python scripts/ulog_tool.py stats LOG.ulg -s vehicle_imu_status@INSTANCE.gyro_vibration_metric --rate 1
python scripts/ulog_tool.py plot LOG.ulg -e "hawkeye_vz=-vehicle_local_position.vz"
python scripts/ulog_tool.py plot LOG.ulg -s vehicle_local_position.z -s vehicle_local_position_setpoint.z --overlay
python scripts/ulog_tool.py stats LOG.ulg -e "vz_error=vehicle_local_position_setpoint.vz-vehicle_local_position.vz" --trend
```

`extract` requires `--output FILE` because sample arrays can be large. Use `--stdout` only for an intentionally small extraction.

`plot` uses separate panels by default. Use `--overlay` only for signals with comparable units.

`accel_vibration_metric` and `gyro_vibration_metric` are EWMA indicators of adjacent integrated-sensor differences. They are relative health indicators, not frequency estimates or physical acceleration amplitudes. Use a high-rate topic such as `sensor_combined` for frequency spectra, and inspect the `primary_instances` section of `recipe vibration` before comparing IMUs or logs.

Known project display mappings are in `signal_aliases.yaml`:

```powershell
python scripts/ulog_tool.py trend LOG.ulg -e hawkeye_vertical_speed
```

## Windows And Alignment

The default is the full common signal window. The tool never silently removes startup or landing data.

```powershell
python scripts/ulog_tool.py recipe vibration LOG.ulg --airborne
python scripts/ulog_tool.py extract LOG.ulg -s vehicle_local_position.z --start 30 --end 80 --rate 20
python scripts/ulog_tool.py compare log46.ulg log86.ulg -s vehicle_local_position.vz --airborne --trend
```

Use `--interpolation previous` for modes, flags, and counters. Use `linear` for continuous signals.

## Compare And Recipes

```powershell
python scripts/ulog_tool.py compare log46.ulg log86.ulg -e hawkeye_vertical_speed --airborne --trend
python scripts/ulog_tool.py compare-signals LOG.ulg --cause vehicle_local_position.vz --effect vehicle_thrust_setpoint.xyz[2] --max-lag 2
python scripts/ulog_tool.py compare log46.ulg log86.ulg -s vehicle_local_position.vz --verbose
python scripts/ulog_tool.py compare-signals LOG.ulg --cause SIGNAL_A --effect SIGNAL_B --all-lags
python scripts/ulog_tool.py transfer LOG.ulg --cause SIGNAL_A --effect SIGNAL_B --rate 1 --at-hz 0.2
python scripts/ulog_tool.py recipe vertical LOG.ulg --full
python scripts/ulog_tool.py recipe gps-ekf LOG.ulg --full
python scripts/ulog_tool.py recipe vibration LOG.ulg --airborne
python scripts/ulog_tool.py recipe attitude-tracking LOG.ulg --start 20 --end 50 --rate 20
python scripts/ulog_tool.py recipe horizontal-control LOG.ulg --start 20 --end 40 --rate 10
```

`stats`, `trend`, `plot`, `recipe`, and `compare` omit sampling diagnostics unless `--verbose` is present. `compare-signals` reports only the strongest lag by default; use `--all-lags` for the full correlation table. If there are paired-signal gaps, it uses only the longest continuous run and reports omitted samples. Cross-correlation is not evidence of causality, and signals must be in compatible coordinate frames before comparing them.

Recipes do not conclude why a pattern occurs. The `vibration` recipe enumerates every IMU instance, reports primary-device changes within the selected window, clipping and errors, and actual raw-topic sampling rates. `attitude-tracking` calculates the shortest quaternion rotation error plus body-rate errors; its shortest-angle error treats `q` and `-q` as the same attitude. `horizontal-control` separates upstream trajectory velocity, the final internal velocity setpoint minus trajectory feed-forward, velocity error, acceleration setpoint, and attitude setpoint. It reports both initial gains and the gains effective at the selected window start, plus any logged in-window horizontal-gain change. Deprecated legacy stats aliases remain available for existing consumers.

For repeated oscillation and large-log checks, use the reusable commands instead of temporary scripts:

```powershell
python scripts/ulog_tool.py spectrum log46.ulg log86.ulg -s vehicle_angular_velocity.xyz[0] --airborne --rate 45 --low-hz 0.5 --high-hz 20
python scripts/ulog_tool.py health log89.ulg --max-file-mib 128
```

`spectrum` reports dominant frequency and band RMS for each selected log and signal. It rejects requests that exceed the source Nyquist rate or would create a spectrum from upsampled sparse data. `health` reports CPU and RAM statistics, dropout totals, and low-stack warnings found in logged messages.

Use `transfer` for gain, phase, and coherence between aligned cause/effect signals. Supply one or more `--at-hz` frequencies. It rejects any requested grid faster than the slowest source by default; choose a lower `--rate`, or use `--allow-upsampling` only when you accept that interpolation adds no bandwidth. It always limits `--at-hz` below the raw-source Nyquist. Treat transfer as compatibility evidence only: cross-spectrum and coherence do not establish causality. Choose windows by flight mode or setpoint phase; do not combine a trajectory transition and a settled hold into one RMS or transfer estimate.

Use temporary code only when the requested operation cannot be expressed with these commands. Keep that code outside the Skill and state why the CLI was insufficient.
