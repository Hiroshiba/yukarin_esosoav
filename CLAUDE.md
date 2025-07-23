# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、LibriTTSなどのデータセットを使用して音素ラベルとアクセント情報（ストレス情報）から母音ごとのピッチの高さを対数F0で予測するための機械学習フレームワークです。PyTorchベースで実装されており、多話者対応のF0予測を行います。

## 主なコンポーネント

### 設定管理
- `src/hiho_pytorch_base/config.py`: Pydantic BaseModelベースの設定管理
  - `DataFileConfig`: データファイルパス設定（feature_vector_pathlist_path、feature_variable_pathlist_path、target_vector_pathlist_path、target_scalar_pathlist_path、speaker_dict_path、root_dir）
  - `DatasetConfig`: データセット設定（train_file、valid_file、test_num等）
  - `NetworkConfig`: ネットワーク設定（feature_vector_size、feature_variable_size、hidden_size、target_vector_size、speaker_size、speaker_embedding_size）
  - `TrainConfig`: 学習設定（batch_size、optimizer、scheduler、use_gpu等）
  - `ProjectConfig`: プロジェクト設定（name、tags、category）

### 学習システム
- `scripts/train.py`: 独自実装のPyTorch学習ループ
  - `train()`: 設定から学習プロセスを実行
  - TensorBoard統合
  - torch.amp（Automatic Mixed Precision）対応
  - 学習済みモデル・スナップショット保存
  - エポックベースの学習スケジューリング

### データ処理
- `src/hiho_pytorch_base/data/data.py`: F0予測用データ構造
  - `InputData`: 音素特徴量、可変長特徴量、F0ターゲット、スカラー値、話者ID
  - `OutputData`: Tensor変換後のデータ構造
  - `preprocess()`: データ前処理（可変長特徴量の平均化等）
- `src/hiho_pytorch_base/dataset.py`: データセット処理
  - pathlistファイル方式でのファイル管理
  - ステムベースのファイル対応付け
  - 遅延読み込み対応（LazyInputData）
  - データタイプ別ディレクトリ管理（feature_vector/、feature_variable/、target_vector/、target_scalar/）

### モデル・ネットワーク
- `src/hiho_pytorch_base/network/predictor.py`: F0予測ネットワーク
  - `Predictor`: 可変長特徴量処理と話者埋め込み対応
  - マルチヘッド出力（vector_head、scalar_head）
  - dropout、ReLU活性化を含む3層MLP
- `src/hiho_pytorch_base/model.py`: 損失計算
  - `Model`: 分類・回帰両方の損失計算
  - cross_entropyとMSE損失の組み合わせ
  - 精度計算機能

### 推論・生成
- `src/hiho_pytorch_base/generator.py`: 学習済みモデルからのF0予測
- `scripts/generate.py`: 推論スクリプト

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

## データ形式

### パスリスト形式
データファイルへのパスをroot_dirからの相対パスで記載：
```
feature_vector/utterance_001.npy
feature_vector/utterance_002.npy
feature_vector/utterance_003.npy
```

### ディレクトリ構造
```
data/
├── feature_vector/     # 音素特徴量
├── feature_variable/   # 可変長特徴量（アクセント情報等）
├── target_vector/      # F0予測ターゲット
├── target_scalar/      # スカラー値
└── speaker_dict.json   # 話者辞書
```

### 話者辞書形式
```json
{
  "speaker_001": ["utterance_001", "utterance_004", "utterance_007"],
  "speaker_002": ["utterance_002", "utterance_005", "utterance_008"],
  "speaker_003": ["utterance_003", "utterance_006", "utterance_009"]
}
```

## 依存関係

### メイン依存関係
- numpy>=2.3.1
- torch>=2.7.1
- pydantic>=2.11.7
- pyyaml>=6.0.2
- tqdm>=4.67.1
- torch-optimizer>=0.3.0

### 開発依存関係
- pytest>=8.4.1
- ruff>=0.12.2
- tensorboard>=2.19.0
- wandb>=0.21.0

## 開発ガイドライン

### コーディング規約
- **フォーマッター**: ruffを使用
- **型アノテーション**: 全ての関数で必須
- **デフォルト値**: 原則として関数引数にデフォルト値を設定しない
- **キーワード引数**: ネットワークのforward methodではキーワード引数のみを使用
- **コメント**: 最小限にし、自明なコメントは避ける

### データ設計
- **ステムベース対応**: 同じサンプルのファイルは拡張子を除いて同じ名前
- **pathlist方式**: root_dirからの相対パスでファイル管理
- **データタイプ別ディレクトリ**: 各データタイプごとに独立したディレクトリ

### 学習システム
- **エポックベース**: スケジューラーはエポック単位で実行
- **マルチタスク学習**: 分類（target_vector）と回帰（target_scalar）を同時学習
- **多話者対応**: 話者埋め込みを使用した多話者F0予測

## 注意事項

- **データ形式**: F0予測のためのマルチタイプデータ（feature_vector, feature_variable, target_vector, target_scalar）を使用
- **話者管理**: speaker_dict.jsonで話者IDとラベルを管理
- **可変長データ**: feature_variableは可変長データとして処理される
- **パスリスト**: root_dirからの相対パス形式でファイルパスを管理する

---

@docs/設計.md
@docs/コーディング規約.md
@.claude/hiho.md
