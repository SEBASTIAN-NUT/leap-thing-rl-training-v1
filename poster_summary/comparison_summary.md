# RL パラメータ最適化: 3 フェーズ比較

| フェーズ | パラメータ | RMSE_vx (100%) | RMSE_yaw (100%) | 説明 |
|---------|----------|---|---|---|
| Phase 1: sigma_lin=0.025 | {'phase': 'Phase 1 (sigma_lin sweep)', 'reward_config.tracking_sigma_lin': 0.025, 'reward_config.tracking_sigma_ang': 1.0} | 0.1205 | 0.4593 | 線速度追従の感度パラメータ最適化。vx 追従 RMSE=0.0515 |
| Phase 2: sigma_ang=2.0 | {'phase': 'Phase 2 (sigma_ang sweep)', 'reward_config.tracking_sigma_lin': 0.025, 'reward_config.tracking_sigma_ang': 2.0} | 0.1728 | 1.0744 | 角速度追従の感度パラメータ最適化。yaw 追従 RMSE=0.4584 |
| Phase 3: alive=0.5 | {'phase': 'Phase 3 (alive sweep)', 'reward_config.scales.alive': 0.5, 'reward_config.scales.tracking_lin_vel': 3.0, 'reward_config.scales.tracking_ang_vel': 1.0, 'reward_config.tracking_sigma_lin': 0.025, 'reward_config.tracking_sigma_ang': 2.0} | 0.1500 | 1.0002 | 生存報酬と追従のバランス最適化。完全統合パラメータ |

## 詳細

### Phase 1: sigma_lin=0.025
**説明**: 線速度追従の感度パラメータ最適化。vx 追従 RMSE=0.0515

**パラメータ**:
- `phase`: Phase 1 (sigma_lin sweep)
- `reward_config.tracking_sigma_lin`: 0.025
- `reward_config.tracking_sigma_ang`: 1.0

**ファイル**: `sigma_lin_0p025/`
- `metadata.json`: パラメータと評価指標
- `time_series.csv`: 推論時系列データ (cmd vs inst)

### Phase 2: sigma_ang=2.0
**説明**: 角速度追従の感度パラメータ最適化。yaw 追従 RMSE=0.4584

**パラメータ**:
- `phase`: Phase 2 (sigma_ang sweep)
- `reward_config.tracking_sigma_lin`: 0.025
- `reward_config.tracking_sigma_ang`: 2.0

**ファイル**: `p2_siggang_2p0/`
- `metadata.json`: パラメータと評価指標
- `time_series.csv`: 推論時系列データ (cmd vs inst)

### Phase 3: alive=0.5
**説明**: 生存報酬と追従のバランス最適化。完全統合パラメータ

**パラメータ**:
- `phase`: Phase 3 (alive sweep)
- `reward_config.scales.alive`: 0.5
- `reward_config.scales.tracking_lin_vel`: 3.0
- `reward_config.scales.tracking_ang_vel`: 1.0
- `reward_config.tracking_sigma_lin`: 0.025
- `reward_config.tracking_sigma_ang`: 2.0

**ファイル**: `p3_alive_0p5/`
- `metadata.json`: パラメータと評価指標
- `time_series.csv`: 推論時系列データ (cmd vs inst)

