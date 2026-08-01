# CPG Locomotion Project

Central Pattern Generator (CPG) による LEAP Hand ロボット歩行制御

## 概要

- **Design**: Stage 2 - Shared IK (MCP+PIP coupled, independent DIP)
- **Contact**: palm_floor_pad (80×110×4mm box) + 3 fingertips
- **Validated Performance**: 1.6-3.3 mm/cycle (push=0.02m, lift=0.03m, freq=1.0Hz)

## ファイル構成

- `leap_thing_cpg.xml` - Robot model (validated)
- `cpg_record.py` - Data recording (headless)
- `cpg_viewer.py` - Interactive simulator
- `cpg_run_final_com.csv` - COM tracking
- `cpg_run_final_if_finger.csv` - Fingertip trajectory
- Videos and trajectory analysis

## 次フェーズ

Parametric sweep と multi-environment verification 後に RL 統合を検討
