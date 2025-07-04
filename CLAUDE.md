# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、機械学習コードを使いやすく、新しく書きやすくするための汎用的な機械学習フレームワークです。PyTorchベースでneural networkの学習・推論を行うためのコードが含まれています。

## 主なコンポーネント

### 設定管理
- `src/hiho_pytorch_base/config.py`: データクラスベースの設定管理
  - `DatasetConfig`: データセット設定（feature_glob、target_glob、test_num等）
  - `NetworkConfig`: ネットワーク設定（実装は空のプレースホルダー）
  - `ModelConfig`: モデル設定（実装は空のプレースホルダー）
  - `TrainConfig`: 学習設定（batch_size、optimizer、use_gpu等）
  - `ProjectConfig`: プロジェクト設定（name、tags、category）

### 学習システム
- `src/hiho_pytorch_base/trainer.py`: pytorch-trainerベースの学習システム（現在は依存関係が削除されているため動作しない）
  - `create_trainer()`: 設定からTrainerオブジェクトを作成
  - TensorBoard、W&B統合
  - AMP（Automatic Mixed Precision）対応
  - スナップショット保存

### データ処理
- `src/hiho_pytorch_base/dataset.py`: データセット処理
  - `FeatureTargetDataset`: numpy配列からのデータセット作成
  - `LazyInputData`: 遅延読み込み対応
  - globパターンでのファイル読み込み

### モデル・ネットワーク
- `src/hiho_pytorch_base/model.py`: モデル定義
  - `Model`: メインモデルクラス（cross_entropy loss、accuracy計算）
  - `Networks`: ネットワークコンポーネント管理
- `src/hiho_pytorch_base/network/predictor.py`: 予測器の実装

### 推論・生成
- `src/hiho_pytorch_base/generator.py`: 学習済みモデルからの推論
- `scripts/generate.py`: 推論スクリプト

### テスト
- `tests/generate_test_data.py`: テストデータ生成コード（10サンプルのfeature/target .npyファイル）
- `tests/conftest.py`: pytest fixtures（テストデータ自動生成機能）
- `tests/test_train.py`: 学習システムのテスト（現在はpytorch-trainer削除のため無効化）

## 主要なファイル

### 学習実行
```bash
uv run python train.py <config_yaml_path> <output_dir>
```

### 推論実行
```bash
uv run python scripts/generate.py --model_dir <model_dir> --output_dir <output_dir> [--use_gpu]
```

### テスト実行
```bash
uv run pytest tests/
```

### 開発環境セットアップ
```bash
uv sync
```

### Ruffでコードフォーマット
```bash
uv run ruff check --fix
uv run ruff format
```

## 設定ファイル形式

YAML形式で設定を管理：
```yaml
dataset:
  feature_glob: "/path/to/feature-npy/*.npy"
  target_glob: "/path/to/target-npy/*.npy"
  test_num: 100
  seed: 0

network: {}

model: {}

train:
  batch_size: 100
  log_iteration: 1000
  eval_iteration: 100000
  stop_iteration: 100000
  optimizer:
    name: "adam"
    lr: 0.001

project:
  name: null
```

## 現在の依存関係

### メイン依存関係 (pyproject.toml)
- numpy>=2.3.1
- torch>=2.7.1
- pyyaml>=6.0.2
- tqdm>=4.67.1

### 開発依存関係 (pyproject.toml [dependency-groups])
- pytest>=8.4.1
- ruff>=0.12.2
- tensorboard>=2.19.0
- wandb>=0.21.0

### パッケージ管理
- UVを使用してpyproject.tomlベースで依存関係を管理
- 最新バージョンの依存関係を使用（PyTorch 2.7.1等）

## 完了した改善点

1. ✅ **UV移行**: pyproject.tomlベースのパッケージ管理に移行完了
2. ✅ **PyTorch更新**: PyTorch 2.7.1に更新完了
3. ✅ **SRCレイアウト**: コードをsrc/hiho_pytorch_base/ディレクトリに移行完了
4. ✅ **Ruff導入**: コードフォーマッターとリンターを導入
5. ✅ **テストデータ生成**: テスト用データ生成コードとpytest fixtures作成完了
6. ✅ **インポートパス更新（部分）**: config.py, dataset.py, trainer.py, model.py, network/predictor.pyで`library` → `hiho_pytorch_base`に更新完了
7. ✅ **.gitignore更新**: GitHub公式Python用テンプレートベースに更新完了

## 今後の作業

1. **インポートパス更新（残り）**: 残りのファイルで`library` → `hiho_pytorch_base`への更新が必要
2. **pytorch-trainer代替**: pytorch-trainerが削除されているため、torch.distributedまたは他の学習ライブラリへの移行が必要
3. **Ruffコード修正**: 143個のエラーを修正する必要（主にdocstring、unused imports等）
4. **Docker更新**: Dockerfileを最新のPyTorchベースイメージに更新
5. **テストシステム復活**: pytorch-trainer代替実装後、test_train.pyを復活させる

## 開発ガイドライン

### 参考プロジェクト
- `../yukarin_sosoa`、`../yukarin_sosfd`、`../accent_estimator`のコードを参考にする
- これらのプロジェクトのどれかに実装があれば、それを真似するようにする

### コーディング規約
- フォーマッターはruffを使用する

## 注意事項

- `src/hiho_pytorch_base/dataset.py:43`にtypoがあります（`input`は`data`であるべき）
- `src/hiho_pytorch_base/model.py:41`で`self.tail`メソッドが定義されていません
- NetworkConfigとModelConfigは現在プレースホルダーのため、実際のネットワーク・モデル設定が必要です
- **重要**: pytorch-trainerが削除されているため、学習システム（trainer.py、model.py）は現在動作しません
- 一部のファイルで古いインポートパス（`from library.xxx`）がまだ残っている可能性があります
- Ruffによるコードチェックで143個のエラーが検出されているため、修正が必要
- tests/test_train.pyは現在無効化されています（pytorch-trainer削除のため）