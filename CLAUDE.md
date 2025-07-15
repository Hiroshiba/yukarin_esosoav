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
- `train.py`: 独自実装のPyTorch学習ループ
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
- `tests/test_train.py`: 学習システムの統合テスト（train.py直接実行・全7テスト実装済み）

## 主要なファイル

### 学習実行
```bash
uv run python train.py <config_yaml_path> <output_dir>
```

### 推論実行
```bash
PYTHONPATH=. uv run python scripts/generate.py --model_dir <model_dir> --output_dir <output_dir> [--use_gpu]
```

### テスト実行
```bash
uv run pytest tests/ -sv
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
  train:
    feature_vector_pathlist_path: "/path/to/feature_vector_pathlist.txt"
    feature_variable_pathlist_path: "/path/to/feature_variable_pathlist.txt"
    target_vector_pathlist_path: "/path/to/target_vector_pathlist.txt"
    target_scalar_pathlist_path: "/path/to/target_scalar_pathlist.txt"
    root_dir: "/path/to/data"
  valid:  # optional
    feature_vector_pathlist_path: "/path/to/valid/feature_vector_pathlist.txt"
    feature_variable_pathlist_path: "/path/to/valid/feature_variable_pathlist.txt"
    target_vector_pathlist_path: "/path/to/valid/target_vector_pathlist.txt"
    target_scalar_pathlist_path: "/path/to/valid/target_scalar_pathlist.txt"
    root_dir: "/path/to/valid_data"
  test_num: 100
  eval_times_num: 1
  seed: 0

network:
  feature_vector_size: 128
  feature_variable_size: 64
  hidden_size: 256
  target_vector_size: 10
  speaker_size: 10
  speaker_embedding_size: 16

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
  scheduler:
    name: "warmup"
    warmup_steps: 100
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
    - utility/pytorch_utility.py（make_optimizer、make_scheduler、collate_list、collate_dataclass等）
    - 新しいPyTorch API対応（torch.amp、WarmupLR等）
    - dataclassベースのcollateシステム実装
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
18. ✅ **インポートパス更新完了**: 全ファイルで`library` → `hiho_pytorch_base`への更新完了
    - scripts/generate.py のインポートパス修正完了
    - 残存していたlibraryインポートを全て修正
19. ✅ **Pydantic設定システム移行**: dataclassからPydantic BaseModelへの移行完了
    - pydantic>=2.11.7 依存関係追加
    - 全configクラス（DatasetFileConfig、DatasetConfig、NetworkConfig等）をBaseModelに変更
    - model_validate()、model_dump(mode="json")による適切なシリアライゼーション
    - PathオブジェクトのYAML変換問題解決
    - 全テスト通過確認
20. ✅ **テストシステム復活**: train.pyベース実装に合わせたtest_train.py復活完了
    - train.pyのtrain関数を直接テスト
    - 実際の学習プロセス実行テスト（test_train_simple_epochs）
    - TensorBoardログ、predictorモデル、snapshotファイル生成確認
    - 全5テスト通過確認
21. ✅ **dataclassベースcollateシステム実装**: TypedDictからdataclassへの移行完了
    - DatasetOutput・BatchOutputをdataclass化（feature_vector、feature_variable、target_vector、target_scalar）
    - 可変長データ対応（feature_variableをList[Tensor]として扱い）
    - type annotation基づくcollate_dataclass関数実装
    - マルチタスク学習対応（分類+回帰）のModel forward method実装
22. ✅ **ディレクトリベースデータ管理への移行**: ステムベースファイル管理実装完了
    - 参照プロジェクト（yukarin_sosoa、yukarin_sosfd、accent_estimator）パターン適用
    - 7文字除去方式からステム（Path.stem）ベース対応付けに変更
    - データタイプ別ディレクトリ構造（feature_vector/、feature_variable/、target_vector/、target_scalar/）
    - テストデータ生成システムの簡略化（同一ファイル名で各ディレクトリに保存）
    - pathlistファイル生成・テストコードの新構造対応
23. ✅ **NetworkConfigリファクタリング**: パラメーター名の分かりやすさとコード整理完了
    - パラメーター名変更（input_size→feature_vector_size、variable_feature_size→feature_variable_size、vector_output_size→target_vector_size）
    - 使われていないoutput_sizeパラメーターを削除
    - Predictorクラスから不要なプロパティ保存を削除（ネットワーク構築にのみ使用）
    - 引数順序を論理的に整理（feature_vector_size、feature_variable_size、hidden_size、target_vector_size）
    - 関連ファイル（create_predictor関数、test_train.py、設定ファイル）の追従完了
24. ✅ **パスリストバグ修正**: dataset.pyのTODO箇所を修正完了
    - `get_data_paths`関数から不要な`/first_data_type`パスを削除
    - pathlistファイルがroot_dirからの相対パス形式を正しく処理するよう修正
    - テストユーティリティのpathlist生成を正しい形式（`feature_vector/0.npy`等）に更新
    - docs/memo.mdにパスリストの設計仕様を詳細に記載
    - 全テスト通過確認（7テスト）

## Docker設計思想

このプロジェクトのDockerfileは、実行環境の提供に特化した設計を採用しています：

- **環境のみ提供**: Dockerfileは依存関係とライブラリのインストールのみを行い、学習コードや推論コードは含みません
- **Git Clone前提**: 実際の利用時は、コンテナ内でGit cloneを実行してコードを取得することを想定しています
- **最新依存関係**: 参照プロジェクト（yukarin_sosoa、yukarin_sosfd、accent_estimator）に準拠し、最新のCUDA/PyTorchベースイメージを使用
- **音声処理対応**: libsoundfile1-dev、libasound2-dev等の音声処理ライブラリの整備方法をコメント等で案内
- **UV使用**: pyproject.tomlベースの依存関係管理にUVを使用し、高速なパッケージインストールを実現

### 使用例
```bash
# コンテナ起動
docker build -t hiho-pytorch-base .
docker run -it --gpus all hiho-pytorch-base

# コンテナ内での作業
git clone https://github.com/your-username/your-project.git
cd your-project
uv run python train.py config.yaml output/
```

## 今後の作業

1. **HDF5対応**: accent_estimatorのようなHDF5データセット対応
2. **話者IDマッピング**: 多話者学習対応

## 開発ガイドライン

### 参考プロジェクト
- `../yukarin_sosoa`、`../yukarin_sosfd`、`../accent_estimator`のコードを参考にする
- これらのプロジェクトのどれかに実装があれば、それを真似するようにする
- **例外**: スケジューラーの実行タイミングは参照プロジェクトと異なり、エポックベースを採用

### コーディング規約
- **フォーマッター**: ruffを使用する
  ```bash
  uv run ruff check --fix
  uv run ruff format
  ```
- **型アノテーション**: 全ての関数の引数と返り値に型アノテーションを必ず記述する
  - 従来のPython型（`list`、`dict`）を使用し、`List[type]`、`Dict[str, type]`は使用しない
  - 複雑な型は`typing.Any`を使用
  - 例: `def function(param: str, data: dict[str, Any]) -> None:`
- **互換性不要**: このプロジェクトは基準となるフレームワークのため、レガシー互換性コードは書かない
- **デフォルト値禁止**: 原則として関数・メソッドの引数にデフォルト値を設定しない
  - デフォルト値はメンテナンス性を下げ、意図しない動作の原因となる
  - 特にnetwork/predictor.py等のコアコンポーネントでは厳格に禁止（Predictorクラスのforwardメソッドなど）
  - 一時的な迂回や問題回避のためのデフォルト値追加は絶対に禁止
  - 例外的に許可される場合：
    - 公開ライブラリのユーザー向けAPI
    - 十分に設計・検討されたシグネチャー（レビューが必要）
  - 内部実装では全ての引数を明示的に渡すこと
  - 適切な設計を行ってから実装すること（デフォルト値で迂回しない）

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