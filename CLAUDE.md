# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、LibriTTSなどのデータセットを使用して音素ラベルと音素ごとのピッチ（対数F0）から音声波形を直接生成するための機械学習フレームワークです。PyTorchベースで実装されており、音響特徴量予測とボコーダーを統合した多話者対応のent-to-end学習を行います。

### 音響特徴量予測とボコーダーの同時学習アプローチ

Conformerエンコーダとボコーダーの統合学習方式を採用：
- 音素ID列と音素ごとのピッチ（対数F0）を組み合わせた埋め込みをConformerエンコーダに入力
- 無声時は対数F0=0として入力
- 各時刻の文脈特徴を抽出し、音響特徴量を中間表現として予測
- ボコーダーで音響特徴量から音声波形を生成
- GAN損失と音響特徴量損失を組み合わせて同時学習
- 学習時は、音響特徴量予測は全体で行うが、波形はメモリ効率のため小さい範囲を切り出して学習

## 主なコンポーネント

以下の主要コンポーネントがあります。
`hiho_pytorch_base`内部のモジュール同士は必ず相対インポートで参照します。

### 設定管理 (`src/hiho_pytorch_base/config.py`)
```python
DataFileConfig:     # ファイルパス設定
DatasetConfig:      # データセット分割設定
NetworkConfig:      # ネットワーク構造設定
ModelConfig:        # モデル設定
TrainConfig:        # 学習パラメータ設定
ProjectConfig:      # プロジェクト情報設定
```

### 学習システム (`scripts/train.py`)
- PyTorch独自実装の学習ループ
- GAN学習
- TensorBoard/W&B統合
- torch.amp（Automatic Mixed Precision）対応
- エポックベーススケジューラー対応
- スナップショット保存・復旧機能

### データ処理 (`src/hiho_pytorch_base/dataset.py`)
- 遅延読み込みによるメモリ効率化
- dataclassベースの型安全なデータ構造
- train/test/eval/valid の4種類データセット対応
- pathlistファイル方式によるファイル管理
- stemベース対応付けで異なるデータタイプを自動関連付け
- 多話者学習対応（JSON形式の話者マッピング）

### ネットワーク (`src/hiho_pytorch_base/network/predictor.py`)
- エンドツーエンド音声生成
- 固定長・可変長データの統一処理
- 音素エンベディング後のLinear層4層による特徴変換
- 音響特徴量予測とボコーダーの統合

#### Conformerベースアーキテクチャ（予定）
- **入力**: 音素ID列 + 音素ごとのピッチ（対数F0）列（全音素数の長さ）
- **エンコーダ**: Conformerエンコーダによる文脈特徴抽出
- **音響特徴量予測**: 各音素位置の音響特徴量を予測
- **ボコーダー**: 音響特徴量から音声波形を生成

### 推論・生成
- `src/hiho_pytorch_base/generator.py`: 音声波形生成ジェネレーター
- `scripts/generate.py`: 音声波形生成実行スクリプト

### テストシステム
- 自動テストデータ生成
- エンドツーエンドテスト
- 統合テスト

## 使用方法

### 学習実行
```bash
uv run -m scripts.train <config_yaml_path> <output_dir>
```

### 推論実行
```bash
uv run -m scripts.generate --model_dir <model_dir> --output_dir <output_dir> [--use_gpu] [--num_files N]
```

### データセットチェック
```bash
uv run -m scripts.check_dataset <config_yaml_path> [--trials 10]
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

## 技術仕様

### 設定ファイル
- **形式**: YAML
- **管理**: Pydanticによる型安全な設定

### 主な依存関係
- **Python**: 3.12+
- **PyTorch**: 2.7.1+
- **NumPy**: 2.2.5+
- **Pydantic**: 2.11.7+
- **librosa**: 0.11.0+（音声処理）
- その他詳細は`pyproject.toml`を参照

### パッケージ管理
- **uv**による高速パッケージ管理
- **pyproject.toml**ベースの依存関係管理

### データ設計
- **ステムベース対応**: 同じサンプルのファイルは拡張子を除いて同じ名前
- **pathlist方式**: root_dirからの相対パスでファイル管理
- **データタイプ別ディレクトリ**: 各データタイプごとに独立したディレクトリ

- **環境のみ提供**: Dockerfileは依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone前提**: 実際の利用時は、コンテナ内でGit cloneを実行してコードを取得することを想定しています
- **音声処理対応**: libsoundfile1-dev、libasound2-dev等の音声処理ライブラリの整備方法をコメント等で案内
- **uv使用**: pyproject.tomlベースの依存関係管理にuvを使用し、高速なパッケージインストールを実現

## フォーク時の使用方法

このフレームワークはフォークして別プロジェクト名でパッケージ化することを想定しています。

### ディレクトリ構造の維持

フォーク後も `src/hiho_pytorch_base/` ディレクトリ名はそのまま維持してください。
ライブラリ内部は相対インポートで実装されているため、ディレクトリ名を変更する必要はありません。

### 拡張例

このフレームワークを拡張する際の参考：

1. **新しいネットワークアーキテクチャ**: `network/`ディレクトリに追加
2. **カスタム損失関数**: `model.py`の拡張
3. **異なるデータ形式**: データローダーの拡張

**注意**: フォーク前からある汎用関数の関数名やdocstringは変更してはいけません。
追従するときにコンフリクトしてしまうためです。

### パッケージ名の変更方法

フォーク先で別のパッケージ名（例: `my_ml_project`）として配布する場合、`pyproject.toml` を以下のように変更します：

```toml
[tool.hatch.build.targets.wheel.sources]
"src/hiho_pytorch_base" = "my_ml_project"
```

---

@docs/設計.md
@docs/コーディング規約.md
@.claude/hiho.md
