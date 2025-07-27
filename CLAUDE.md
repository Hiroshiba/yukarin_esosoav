# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、LibriTTSなどのデータセットを使用して音素ラベルとアクセント情報（ストレス情報）から母音ごとのピッチの高さを対数F0で予測するための機械学習フレームワークです。PyTorchベースで実装されており、多話者対応のF0予測と有声/無声予測を行います。

### F0予測アプローチ

Conformerエンコーダベースの直接予測方式を採用：
- 音素ID列と母音/ストレスフラグを組み合わせた埋め込みをConformerエンコーダに入力
- 各時刻の文脈特徴を抽出し、回帰ヘッドで全音素のF0を予測
- 母音フラグマスクを適用して母音部分のF0のみを抽出し最終出力とする
- 有声/無声の二値分類ヘッドで有声かどうかを予測

## 主なコンポーネント

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
- TensorBoard/W&B統合
- torch.amp（Automatic Mixed Precision）対応
- エポックベーススケジューラー対応
- スナップショット保存・復旧機能

### データ処理 (`src/hiho_pytorch_base/dataset.py`)
- 4種類のデータタイプの統一処理
- 遅延読み込みによるメモリ効率化
- dataclassベースの型安全なデータ構造
- train/test/eval/valid の4種類データセット対応
- pathlistファイル方式によるファイル管理
- stemベース対応付けで異なるデータタイプを自動関連付け
- 多話者学習対応（JSON形式の話者マッピング）

### ネットワーク (`src/hiho_pytorch_base/network/predictor.py`)
- F0回帰 + vuv分類
- 固定長・可変長データの統一処理
- 音素エンベディング後のLinear層4層による特徴変換
- 2つの出力ヘッド：F0予測用とvuv予測用

#### Conformerベースアーキテクチャ（予定）
- **入力**: 音素ID列 + 母音/ストレスフラグ列（全音素数の長さ）
- **エンコーダ**: Conformerエンコーダによる文脈特徴抽出
- **回帰ヘッド**: 各音素位置のF0値を予測
- **出力処理**: 母音フラグマスクで母音部分のF0のみを抽出

### 推論・生成
- `src/hiho_pytorch_base/generator.py`: 推論ジェネレーター
- `scripts/generate.py`: 推論実行スクリプト

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

- **環境のみ提供**: Dockerfileは依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone前提**: 実際の利用時は、コンテナ内でGit cloneを実行してコードを取得することを想定しています
- **最新依存関係**: 参照プロジェクト（yukarin_sosoa、yukarin_sosfd、accent_estimator）に準拠し、最新のCUDA/PyTorchベースイメージを使用
- **音声処理対応**: libsoundfile1-dev、libasound2-dev等の音声処理ライブラリの整備方法をコメント等で案内
- **uv使用**: pyproject.tomlベースの依存関係管理にuvを使用し、高速なパッケージインストールを実現

## フォーク時の拡張例

このフレームワークを拡張する際の参考：

1. **新しいネットワークアーキテクチャ**: `network/`ディレクトリに追加
2. **カスタム損失関数**: `model.py`の拡張
3. **異なるデータ形式**: データローダーの拡張

### 参考プロジェクト
- 以下のプロジェクトの実装パターンを参考にしている
- `../yukarin_sosoa`、`../yukarin_sosfd`、`../accent_estimator`

---

@docs/設計.md
@docs/コーディング規約.md
@.claude/hiho.md
