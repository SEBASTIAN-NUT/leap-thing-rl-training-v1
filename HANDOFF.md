# LEAP Hand 指先歩行 RL — 引継ぎドキュメント

**担当者:** 飯澤  
**引継ぎ日:** 2026-08-10  
**研究概要:** LEAP Hand（16自由度多指ハンド）の指先で自律歩行する方策をMuJoCo MJX + Brax PPOで強化学習

---

## ディレクトリ構成

    iizawa_workspace/
    ├── thing_project/          ← メインプロジェクト（このフォルダ）
    ├── mujoco_playground/      ← 依存ライブラリ
    ├── mujoco_menagerie/       ← MuJoCoロボットモデル集
    ├── IsaacLab/               ← 参考（本研究では未使用）
    ├── _archive/               ← 旧実験データ（削除不要）
    ├── discord_notifier.py     ← 学習完了Discord通知スクリプト
    ├── test_go1.py             ← Go1 XML動作確認用
    └── start_env.sh            ← venv有効化ショートカット

    thing_project/
    ├── thing_test/             ← 環境定義・学習コード
    │   ├── xmls/               ← MuJoCo XMLモデル（scene_flat_terrain.xml等）
    │   ├── common/             ← 共通ユーティリティ
    │   └── check_command_response.py  ← 推論評価スクリプト（CCR）
    ├── experiments/            ← フェーズ別実験スクリプト
    ├── training/            <- 学習エントリ + 実験スクリプト
    ├── eval_results/           ← CCR評価結果（CSV）
    ├── analysis_output/        ← 報酬曲線CSV・PNG
    ├── poster_summary/         ← ポスター用動画・サマリー
    ├── poster_figures/         ← ポスター用グラフ
    ├── Open_Duck_Playground/   ← 依存サブモジュール（.venv含む）
    ├── Open_Duck_Mini/         ← 参考実装
    └── cpg_project/            ← CPG実装（今後の作業）

---

## 環境セットアップ

    cd ~/Desktop/iizawa/iizawa_workspace/thing_project
    source Open_Duck_Playground/.venv/bin/activate
    # プロンプトが (open-duck-playground) になれば OK

---

## 実験フェーズと結果

### Phase 2a: sigma_lin スイープ（sigma_ang=0.25, alive=0）

| 条件 | 評価 |
|---|---|
| sigma_lin=0.0025 | 狭すぎ、学習困難 |
| sigma_lin=0.005  | 狭すぎ |
| sigma_lin=0.01   | やや狭い |
| sigma_lin=0.025  | 最良（採用） |
| sigma_lin=0.05   | 広すぎ、報酬が飽和 |

→ **sigma_lin=0.025 を採用**

### Phase 2b: sigma_ang スイープ（sigma_lin=0.025, alive=0）

| 条件 | 評価 |
|---|---|
| sigma_ang=0.1 〜 2.0 | グラフに顕著な差なし |

→ **sigma_ang=2.0 を採用**（緩い方が安定傾向）

### Phase 3: alive リワードスイープ（sigma_lin=0.025, sigma_ang=2.0）

| 条件 | チェックポイント |
|---|---|
| alive=0.1 | p3_alive_0p1_20260803_023938 |
| alive=0.2 | p3_alive_0p2_20260803_024512 |
| alive=0.3 | p3_alive_0p3_20260716_090221 |
| alive=0.5 | p3_alive_0p5_20260711_105838 |
| alive=1.0 | p3_alive_1p0_20260715_144958 |

alive は生存ボーナス（正の報酬）。大きいほど転倒回避が強まるが追従が弱まる。

### Phase 5: palm 接触ペナルティ（探索）

- p5_base_palm_20260730: 手のひら接触へのペナルティ実験

---

## 推論の実行（CCR）

    cd ~/Desktop/iizawa/iizawa_workspace/thing_project
    source Open_Duck_Playground/.venv/bin/activate

    # GUIビューワあり
    python -m thing_test.check_command_response \
        -o checkpoints/p3_alive_0p1_20260803_023938/<ONNX_FILE>.onnx

    # ビューワなし、評価のみ
    python -m thing_test.check_command_response \
        -o checkpoints/p3_alive_0p1_20260803_023938/<ONNX_FILE>.onnx \
        --no_viewer --axes vx,yaw --fractions=0.25,0.5,0.75,1.0

---

## 学習の実行

    cd ~/Desktop/iizawa/iizawa_workspace/thing_project
    source Open_Duck_Playground/.venv/bin/activate

    # フェーズ別スイープ（例: Phase 3）
    python experiments/phase3_alive_sweep.py

---

## 今後の作業（先生との方針）

1. alive=0 での再確認 — アライブ報酬なし（ペナルティなし）での基準動作確認
2. Go1 ロボットへの置換 — LEAP Hand から Unitree Go1 XML へ差し替え
3. CPG 統合 — cpg_project/ にCPG実装あり。学習器と接続する
4. home position 変更 — scene_flat_terrain.xml の keyframe 調整
5. 文献調査 — 指先歩行・CPG関連論文の整理

---

## 重要な知見・注意事項

- JAX_DISABLE_JIT が設定されていると学習がコンパイルされず超低速になる。
  確認: echo $JAX_DISABLE_JIT  解除: unset JAX_DISABLE_JIT

- GPU クラッシュ (0xbadf5720): 旧Maxwell GPU (GM107) で高負荷時に発生。
  watch -n 2 nvidia-smi で温度監視推奨。

- チェックポイント形式: Orbax OCDBT (JAX標準) + ONNX（推論用）。
  ONNXファイル名にステップ数が入っている（例: 2026_08_03_025438_10000000.onnx）

- 報酬式: r = scale × exp(-e² / sigma)  e = cmd - meas

- venv: Open_Duck_Playground/.venv/ を使用。uv sync で再構築可能。

