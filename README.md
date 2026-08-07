# ulg_analyze

用于 Codex 的 PX4 `.ulg` 飞行日志分析 skill。日志格式的正式名称是 ULog，文件扩展名是 `.ulg`。

它将常用的日志检查、字段和参数查询、信号提取、统计、趋势、频谱、控制响应、IMU 振动和系统健康检查封装为可复用命令，避免每次分析都临时编写解析脚本。

## 内容

- `SKILL.md`：Codex 的分析规则和命令入口
- `scripts/ulog_tool.py`：命令行工具
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
python scripts/ulog_tool.py spectrum flight.ulg -s vehicle_angular_velocity.xyz[0] --airborne
```

## 更新

本仓库只保存 skill 代码和说明，不保存原始 `.ulg` 日志或分析结果。更新本地 skill：

```powershell
git pull
```

## 日志索引

skill 可选地使用外部 `flight_index.json` 记录日志概述，用于跨日志检索。该 JSON 索引不是数据库引擎，也不保存分析细节；原始日志和索引仍保留在本机。
