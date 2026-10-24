# ulog_analyze

用于 Codex 的 PX4 `.ulg` 飞行日志分析 skill。日志格式的正式名称是 ULog，文件扩展名是 `.ulg`。

它将常用的日志检查、字段和参数查询、信号提取、统计、趋势、频谱、控制响应、IMU 振动和系统健康检查封装为可复用命令，避免每次分析都临时编写解析脚本。

## 内容

- `SKILL.md`：Codex 的分析规则和命令入口
- `scripts/ulog_tool.py`：命令行工具（含采样安全与客观状态时间线）
- `scripts/fifo_tools.py`：原生 IMU FIFO 样本展开工具
- `scripts/flight_catalog.py`：增量维护飞行日志索引
- `references/pyulog.md`：`pyulog` 的使用注意事项
- `tests/`：工具测试

## 依赖

```powershell
python -m pip install numpy scipy pyulog matplotlib
```

绘图命令需要 `matplotlib`，其余命令按实际功能使用 `numpy`、`scipy` 和 `pyulog`。

## 使用示例

```powershell
python scripts/ulog_tool.py inspect flight.ulg
python scripts/ulog_tool.py topics flight.ulg
python scripts/ulog_tool.py recipe vibration flight.ulg --airborne
python scripts/ulog_tool.py spectrum flight.ulg -s vehicle_angular_velocity.xyz[0] --airborne --rate 50 --low-hz 0.5 --high-hz 20
python scripts/ulog_tool.py recipe motors flight.ulg --airborne
python scripts/ulog_tool.py recipe gps-ekf flight.ulg --airborne
python scripts/ulog_tool.py timeline flight.ulg -s vehicle_status.nav_state -s vehicle_land_detected.landed
python scripts/ulog_tool.py native flight.ulg --topic sensor_accel --field x --field y --field z --output accel_native.csv
python scripts/ulog_tool.py fifo flight.ulg --topic sensor_accel_fifo --instance 0 --anchor first --output accel_fifo.csv
```

## 更新

本仓库只保存 skill 代码和说明，不保存原始 `.ulg` 日志或分析结果。更新本地 skill：

```powershell
git pull
```

## 日志索引

skill 可选地使用外部 `flight_index.json` 记录日志概述，用于跨日志检索。该 JSON 索引不是数据库引擎，也不保存分析细节；原始日志和索引仍保留在本机。

## 新版数据处理边界

本仓库仅负责 ULog 数据提取、校验、计算及结构化输出；**不内置根因推断、稳定性评级、固定调参逻辑或 PID 建议**。

- **抗混叠**：`spectrum` 与 `transfer` 的连续信号降采样前做低通，数据时间间隔过度不均匀或连续样本不足时明确拒绝，而不是计算可能有混叠的频谱。离散状态使用前值插值，不滤波。
- **原生采样**：`native` 导出日志所记录的原始 topic 采样；`fifo` 将 `sensor_accel_fifo` / `sensor_gyro_fifo` 中有效样本按逐包 `dt` 和 `scale` 解包，不用 packet 发布频率冒充物理采样频率。
- **时间戳**：`fifo --anchor first|last` 必须明确指定 `timestamp_sample` 对应包内第一个还是最后一个采样点；根据飞控固件/驱动验证约定，不作未经证实的默认假定。
- **六旋翼和其他配置**：`recipe motors` 根据 ULog 格式枚举全部 `actuator_motors.control[i]` 通道，空通道保留为无有效采样，不默认为四旋翼。
- **水平 GPS/EKF**：`recipe gps-ekf` 包含 N/E 速度、水平本地位置及可用创新字段；缺失的字段如实报告，不用零填补。
- **状态分段**：`timeline` 只列出状态或模式的真实变更事件，不推断其原因或是否稳定。

若计算某频段百分比，需要明确分母和类型（均方/功率/方差占比，或 RMS 幅值占比），**不得把均方量百分比误称为幅值百分比**。

运行测试：`python -m unittest discover -s tests -v`（需要安装依赖）。
