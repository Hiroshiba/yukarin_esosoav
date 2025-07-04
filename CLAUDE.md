# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、機械学習コードを使いやすく、新しく書きやすくするための汎用的な機械学習フレームワークです。PyTorchベースでneural networkの学習・推論を行うためのコードが含まれています。

## 主なコンポーネント

### 設定管理
- `src/hiho_pytorch_base/config.py`: データクラスベースの設定管理
  - `DatasetFileConfig`: ファイルパス設定（feature_pathlist_path、target_pathlist_path、root_dir）
  - `DatasetConfig`: データセット設定（train_file、valid_file、test_num等）
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
  - `DatasetInput`: feature/targetデータ構造
  - `LazyDatasetInput`: 遅延読み込み対応
  - `DatasetOutput`: TypedDictによるアウトプット定義
  - `FeatureTargetDataset`: PyTorchデータセット実装
  - `preprocess()`: 前処理関数
  - `_load_pathlist()`: pathlistファイル読み込み
  - `get_datas()`: データ取得関数
  - `create_dataset()`: データセット作成（train/test/eval/valid対応）
  - pathlistファイル方式でのファイル管理（yukarin_sosfd準拠）

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
  train_file:
    feature_pathlist_path: "/path/to/feature_pathlist.txt"
    target_pathlist_path: "/path/to/target_pathlist.txt"
    root_dir: "/path/to/data"
  valid_file:  # optional
    feature_pathlist_path: "/path/to/valid_feature_pathlist.txt"
    target_pathlist_path: "/path/to/valid_target_pathlist.txt"
    root_dir: "/path/to/valid_data"
  test_num: 100
  eval_times_num: 1
  seed: 0

network: {}

model: {}

train:
  batch_size: 100
  eval_batch_size: 10
  log_epoch: 1
  eval_epoch: 10
  snapshot_epoch: 100
  stop_epoch: 1000
  model_save_num: 5
  optimizer:
    name: "adam"
    lr: 0.001
  scheduler: null
  num_processes: 4
  use_gpu: true
  use_amp: true

project:
  name: null
```

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
8. ✅ **dataset.py最新化**: yukarin_sosoa/sosfdを参考に全面的に更新完了
   - pathlist方式への移行（glob方式から変更）
   - DatasetFileConfig導入によるスケーラブルな設定管理
   - DatasetOutput TypedDict化
   - preprocess関数、_load_pathlist関数、get_datas関数追加
   - バリデーションデータ対応（valid_file設定）
   - typo修正（43行目input→data）
9. ✅ **model.py最新化**: yukarin_sosoa/sosfdを参考に全面的に更新完了
   - ModelOutput TypedDict化（loss、accuracy、data_num）
   - reduce_result関数追加（辞書アクセス対応）
   - self.tail問題解決（predictor直接使用に変更）
   - pytorch_trainer依存削除
   - DatasetOutput対応の forward method実装
10. ✅ **generator.py最新化**: yukarin_sosoa/sosfdを参考に更新完了
    - インポートパス修正（library → hiho_pytorch_base）
    - GeneratorOutput クラス導入
    - to_tensor関数追加
    - nn.Module継承のGenerator実装
    - generate/forward メソッド両対応
11. ✅ **network/predictor.py最新化**: 基本的な実装を追加完了
    - シンプルなMLP実装（3層のLinear + ReLU + Dropout）
    - NetworkConfig対応のcreate_predictor関数
    - デフォルトパラメータ設定（input_size=128、hidden_size=256、output_size=10）
12. ✅ **evaluator.py新規作成**: model.pyのlossを評価指標にした実装完了
    - EvaluatorOutput TypedDict化（value、loss、accuracy、data_num）
    - Generator使用の評価システム
    - cross_entropy loss計算とaccuracy計算
    - judge プロパティ追加（"min"/"max"判定用）
13. ✅ **trainer.py削除とtrain.py作成**: pytorch-trainer代替実装完了
    - trainer.pyを削除（参照プロジェクト準拠）
    - train.pyを参照プロジェクト（yukarin_sosoa）ベースで新規作成
    - torch.amp（新しいPyTorch API）使用
    - torch.jit.script による predictor 最適化
    - datasets["eval"] 使用、valid_dataset None 対応
    - evaluator.judge を使用した SaveManager 実装
14. ✅ **utility モジュール実装**: 参照プロジェクト準拠で新規作成
    - utility/train_utility.py（Logger、SaveManager）
    - utility/pytorch_utility.py（make_optimizer、make_scheduler、collate_list等）
    - 新しいPyTorch API対応（torch.amp、WarmupLR等）
15. ✅ **config.py エポックベース移行**: iteration → epoch ベースに更新完了
    - TrainConfig を iteration ベースから epoch ベースに変更
    - model_save_num、scheduler、pretrained_predictor_path 等フィールド追加
    - 参照プロジェクト準拠の設定項目統一
16. ✅ **torch_optimizer パッケージ追加**: uv add で依存関係追加完了
    - torch-optimizer>=0.3.0 追加
    - pytorch-ranger>=0.1.1 も自動インストール
    - RAdam、Ranger等の最新オプティマイザ利用可能
17. ✅ **型整合性修正**: 各種型エラー・実行時エラー対応完了
    - Model コンストラクタ torch.jit.script 対応（nn.Module型に変更）
    - valid_dataset None 対応処理追加
    - target データ型修正（.float() → .long()、cross_entropy用）
    - DatasetOutput/GeneratorOutput 型一貫性確保

## 今後の作業

1. **インポートパス更新（残り）**: 残りのファイルで`library` → `hiho_pytorch_base`への更新が必要
2. **Ruffコード修正**: docstring、unused imports等のエラー修正が必要
3. **Docker更新**: Dockerfileを最新のPyTorchベースイメージに更新
4. **テストシステム復活**: train.py ベース実装に合わせて test_train.py を復活させる

## 開発ガイドライン

### 参考プロジェクト
- `../yukarin_sosoa`、`../yukarin_sosfd`、`../accent_estimator`のコードを参考にする
- これらのプロジェクトのどれかに実装があれば、それを真似するようにする

### コーディング規約
- フォーマッターはruffを使用する
- 型アノテーション：`List[type]`、`Dict[str, type]`、`list[type]`、`dict[str, type]`は使用しない。従来のPython型（list、dict）を使用する
- **互換性不要**: このプロジェクトは基準となるフレームワークのため、レガシー互換性コードは書かない
- **デフォルト値禁止**: network/predictor.py等のコアコンポーネントでは引数にデフォルト値を設定しない（特にPredictorクラス）

## 注意事項

- NetworkConfigとModelConfigは現在プレースホルダーのため、実際のネットワーク・モデル設定が必要です
- **学習システム**: pytorch-trainerは削除され、新しいtrain.pyでネイティブPyTorch学習ループが動作します
- 一部のファイルで古いインポートパス（`from library.xxx`）がまだ残っている可能性があります
- Ruffによるコードチェックでdocstring、unused imports等のエラーが残っている
- tests/test_train.pyは現在無効化されています（train.pyベース実装への移行対応が必要）