# 推奨パラメータセット比較

| パラメータセット | RMSE_vx (100%) | RMSE_yaw (100%) | 特徴 |
|---|---|---|---|
| Phase 3: alive=0.5 | 0.1500 | 1.0002 | 生存報酬と追従のバランスが最適な alive スケール |
| Phase 5: stand_still=-1.0 | 0.1500 | 1.0001 | 停止時のドリフト抑制を追加。yaw 追従性能向上（RMSE_yaw=0.0021） |
| Phase 5: feet_air_time=0.1 | 0.1500 | 1.0001 | 指先の空中時間報酬を追加。歩行リズム改善 |

## 詳細

### Phase 3: alive=0.5
**説明**: 生存報酬と追従のバランスが最適な alive スケール

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

### Phase 5: stand_still=-1.0
**説明**: 停止時のドリフト抑制を追加。yaw 追従性能向上（RMSE_yaw=0.0021）

**パラメータ**:
- `phase`: Phase 5 (Go1 terms - stand_still)
- `reward_config.scales.stand_still`: -1.0
- `from_phase_3`: alive=0.5, trk_lin=3.0, trk_ang=1.0

**ファイル**: `p5_stand_still_m1p0/`
- `metadata.json`: パラメータと評価指標
- `time_series.csv`: 推論時系列データ (cmd vs inst)

### Phase 5: feet_air_time=0.1
**説明**: 指先の空中時間報酬を追加。歩行リズム改善

**パラメータ**:
- `phase`: Phase 5 (Go1 terms - feet_air_time)
- `reward_config.scales.feet_air_time`: 0.1
- `from_phase_3`: alive=0.5, trk_lin=3.0, trk_ang=1.0

**ファイル**: `p5_feet_air_time_0p1/`
- `metadata.json`: パラメータと評価指標
- `time_series.csv`: 推論時系列データ (cmd vs inst)

