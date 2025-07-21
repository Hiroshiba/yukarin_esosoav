# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、機械学習コードを使いやすく、新しく書きやすくするための汎用的な機械学習フレームワークです。PyTorchベースでneural networkの学習・推論を行うためのコードが含まれています。

## 主なコンポーネント

### 設定管理
- `src/hiho_pytorch_base/config.py`: Pydantic BaseModelベースの設定管理
  - `DataFileConfig`: ファイルパス設定（feature_vector_pathlist_path、feature_variable_pathlist_path、target_vector_pathlist_path、target_scalar_pathlist_path、speaker_dict_path、root_dir）
  - `DatasetConfig`: データセット設定（train_file、valid_file、test_num等）
  - `NetworkConfig`: ネットワーク設定（feature_vector_size、feature_variable_size、hidden_size、target_vector_size、speaker_size、speaker_embedding_size）
  - `ModelConfig`: モデル設定（実装は空のプレースホルダー）
  - `TrainConfig`: 学習設定（batch_size、optimizer、scheduler、use_gpu等）
  - `ProjectConfig`: プロジェクト設定（name、tags、category）

### 学習システム
- `scripts/train.py`: 独自実装のPyTorch学習ループ
  - `train()`: 設定から学習プロセスを実行
  - TensorBoard統合
  - torch.amp（Automatic Mixed Precision）対応
  - 学習済みモデル・スナップショット保存

### データ処理
- `src/hiho_pytorch_base/dataset.py`: データセット処理
  - `DatasetInput`: feature/targetデータ構造
  - `LazyDatasetInput`: 遅延読み込み対応
  - `DatasetOutput`: dataclassによるアウトプット定義（feature_vector, feature_variable, target_vector, target_scalar）
  - `BatchOutput`: collate後のバッチデータ構造（可変長データはList[Tensor]）
  - `FeatureTargetDataset`: PyTorchデータセット実装
  - `preprocess()`: 前処理関数
  - `_load_pathlist()`: pathlistファイル読み込み
  - `get_datas()`: データ取得関数（ステムベースのファイル管理）
  - `create_dataset()`: データセット作成（train/test/eval/valid対応）
  - pathlistファイル方式でのファイル管理（参照プロジェクト準拠）
  - **パスリスト形式**: root_dirからの相対パス（`feature_vector/0.npy`等）を記載
  - **ディレクトリ構造**: データタイプ別ディレクトリ管理（feature_vector/, feature_variable/, target_vector/, target_scalar/）
  - **stemベース対応付け**: 同じファイル名（stem）で異なるデータタイプを関連付け

### モデル・ネットワーク
- `src/hiho_pytorch_base/model.py`: マルチタスク学習対応のモデル定義
  - `Model`: メインモデルクラス（分類・回帰両方の損失計算、accuracy計算）
  - `ModelOutput`: dataclassによる出力定義（loss、loss_vector、loss_scalar、accuracy、data_num）
- `src/hiho_pytorch_base/network/predictor.py`: マルチタスク学習対応の予測器実装
  - `Predictor`: 固定長・可変長データ両方を処理し、ベクトル出力・スカラー出力を生成
  - 可変長データ処理（variable_processor）とマルチヘッド出力（vector_head、scalar_head）
  - `create_predictor()`: NetworkConfigから予測器を作成

### 推論・生成
- `src/hiho_pytorch_base/generator.py`: 学習済みモデルからの推論
- `scripts/generate.py`: 推論スクリプト

### テスト
- `tests/test_utils.py`: テストデータ生成ユーティリティ（マルチタイプデータ対応・ディレクトリ構造対応）
  - `setup_data()`: 4タイプデータ生成（feature_vector、feature_variable、target_vector、target_scalar）
  - `create_train_config()`: テスト用設定作成
  - pathlistファイル生成（root_dirからの相対パス形式）
- `tests/conftest.py`: pytest fixtures（テストデータ自動生成機能）
- `tests/test_train.py`: 学習システムの統合テスト（scripts/train.py直接実行・全7テスト実装済み）

## 主要なファイル

### 学習実行
```bash
uv run -m scripts.train <config_yaml_path> <output_dir>
```

### 生成実行
```bash
uv run -m scripts.generate --model_dir <model_dir> --output_dir <output_dir> [--use_gpu]
```

### テスト実行
```bash
uv run pytest tests/ -sv
```

### 開発環境セットアップ
```bash
uv sync
```

### 静的解析とフォーマット
```bash
uv run pyright && uv run ruff check --fix && uv run ruff format
```

## 設定ファイル形式

YAML形式で設定を管理

## 現在の依存関係

### メイン依存関係 (pyproject.toml)
- numpy>=2.3.1
- torch>=2.7.1
- pyyaml>=6.0.2
- tqdm>=4.67.1
- torch-optimizer>=0.3.0

### 開発依存関係 (pyproject.toml [dependency-groups])
- pytest>=8.4.1
- ruff>=0.12.2
- tensorboard>=2.19.0
- wandb>=0.21.0

### パッケージ管理
- uvを使用してpyproject.tomlベースで依存関係を管理
- 最新バージョンの依存関係を使用（PyTorch 2.7.1等）

## Docker設計思想

このプロジェクトのDockerfileは、実行環境の提供に特化した設計を採用しています：

- **環境のみ提供**: Dockerfileは依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone前提**: 実際の利用時は、コンテナ内でGit cloneを実行してコードを取得することを想定しています
- **最新依存関係**: 参照プロジェクト（yukarin_sosoa、yukarin_sosfd、accent_estimator）に準拠し、最新のCUDA/PyTorchベースイメージを使用
- **音声処理対応**: libsoundfile1-dev、libasound2-dev等の音声処理ライブラリの整備方法をコメント等で案内
- **uv使用**: pyproject.tomlベースの依存関係管理にuvを使用し、高速なパッケージインストールを実現

## 今後の作業

1. **HDF5対応**: accent_estimatorのようなHDF5データセット対応

### 参考プロジェクト
- `../yukarin_sosoa`、`../yukarin_sosfd`、`../accent_estimator`のコードを参考にする
- これらのプロジェクトのどれかに実装があれば、それを真似するようにする
- **例外**: スケジューラーの実行タイミングは参照プロジェクトと異なり、エポックベースを採用

## 注意事項

- **NetworkConfig**: マルチタスク学習・多話者学習対応で実装済み（feature_vector_size、feature_variable_size、hidden_size、target_vector_size、speaker_size、speaker_embedding_size）
- **ModelConfig**: 現在プレースホルダーのため、実際のモデル設定が必要です
- **学習システム**: pytorch-trainerは削除され、新しいtrain.pyでネイティブPyTorch学習ループが動作します
- **データ構造**: dataclassベースのマルチタイプデータ（feature_vector, feature_variable, target_vector, target_scalar）を使用
- **ディレクトリ構造**: データタイプ別ディレクトリに同一ファイル名（ステムベース）で保存する方式を採用
- **パスリスト**: root_dirからの相対パス形式（`feature_vector/0.npy`等）でファイルパスを管理
- **スケジューラー**: エポックベースで実行（参照プロジェクトはイテレーションベース）
  - WarmupLRスケジューラーの`warmup_steps`はエポック数として解釈される
  - 設定例: `warmup_steps: 100`（100エポック）

---

@docs/設計.md
@docs/コーディング規約.md
