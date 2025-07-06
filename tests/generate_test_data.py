from pathlib import Path

import numpy as np


def generate_noisy_correlated_data(
    feature_dir: Path,
    target_dir: Path,
    num_samples: int = 300,
    feature_shape: tuple[int, ...] = (64,),
    num_classes: int = 3,
    noise_level: float = 0.3,
    seed: int = 42,
) -> None:
    """
    ノイジーな相関のあるテストデータを生成する

    Args:
        feature_dir: 特徴量ファイルを保存するディレクトリ
        target_dir: ターゲットファイルを保存するディレクトリ
        num_samples: サンプル数
        feature_shape: 特徴量の形状
        num_classes: クラス数
        noise_level: ノイズの強さ (0.0-1.0)
        seed: 乱数シード
    """
    # ディレクトリを作成
    feature_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 乱数シードを設定
    np.random.seed(seed)

    # 各クラスに対応する基本パターンを生成
    class_patterns = []
    for class_idx in range(num_classes):
        # 各クラスに特徴的なパターンを作成
        pattern = np.random.randn(*feature_shape).astype(np.float32)
        # パターンを正規化
        pattern = pattern / np.linalg.norm(pattern)
        class_patterns.append(pattern)

    # データを生成
    for i in range(num_samples):
        # ターゲットクラスをランダムに選択
        target_class = np.random.randint(0, num_classes, dtype=np.int64)

        # 基本パターンを取得
        base_pattern = class_patterns[target_class].copy()

        # 相関のある特徴量を生成（基本パターン + ノイズ）
        correlation_signal = base_pattern * (1.0 - noise_level)
        noise = np.random.randn(*feature_shape).astype(np.float32) * noise_level
        feature = correlation_signal + noise

        # 特徴量を正規化
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        # ファイルに保存（ファイル名を一致させるため番号のみを使用）
        feature_path = feature_dir / f"{i:03d}.npy"
        target_path = target_dir / f"{i:03d}.npy"

        np.save(feature_path, feature)
        np.save(target_path, target_class)

    print(f"Generated {num_samples} noisy correlated test samples")
    print(f"Feature files saved to: {feature_dir}")
    print(f"Target files saved to: {target_dir}")


def generate_test_data(
    feature_dir: Path,
    target_dir: Path,
    num_samples: int = 10,
    feature_shape: tuple[int, ...] = (64,),
    num_classes: int = 3,
    seed: int = 42,
) -> None:
    """
    テスト用のデータセットを生成する関数（後方互換性のため残存）

    Args:
        feature_dir: 特徴量ファイルを保存するディレクトリ
        target_dir: ターゲットファイルを保存するディレクトリ
        num_samples: サンプル数
        feature_shape: 特徴量の形状
        num_classes: クラス数
        seed: 乱数シード
    """
    # 新しいノイジー相関データ生成機能を使用
    generate_noisy_correlated_data(
        feature_dir=feature_dir,
        target_dir=target_dir,
        num_samples=num_samples,
        feature_shape=feature_shape,
        num_classes=num_classes,
        noise_level=0.3,
        seed=seed,
    )


def create_pathlist_files(
    feature_dir: Path,
    target_dir: Path,
    base_dir: Path,
    train_count: int = 100,
    valid_count: int = 100,
    seed: int = 42,
) -> dict[str, Path]:
    """
    pathlistファイルを生成する

    Args:
        feature_dir: 特徴量ファイルのディレクトリ
        target_dir: ターゲットファイルのディレクトリ
        base_dir: pathlistファイルを保存するディレクトリ
        train_count: 訓練データ数
        valid_count: バリデーションデータ数
        seed: 乱数シード

    Returns
    -------
        dict: pathlistファイルのパス辞書
    """
    np.random.seed(seed)

    # 全ファイルのリストを取得
    feature_files = sorted(feature_dir.glob("*.npy"))
    target_files = sorted(target_dir.glob("*.npy"))

    total_files = len(feature_files)
    total_needed = train_count + valid_count

    if total_files < total_needed:
        print(f"Warning: Only {total_files} files available, but {total_needed} needed")

    # インデックスをシャッフル
    indices = np.arange(total_files)
    np.random.shuffle(indices)

    # データを分割
    train_indices = indices[:train_count]
    valid_indices = indices[train_count : train_count + valid_count]

    # pathlistファイルを生成
    pathlist_files = {}

    for split_name, split_indices in [
        ("train", train_indices),
        ("valid", valid_indices),
    ]:
        if len(split_indices) == 0:
            continue

        feature_pathlist = base_dir / f"{split_name}_feature_pathlist.txt"
        target_pathlist = base_dir / f"{split_name}_target_pathlist.txt"

        # pathlistファイルを作成
        with open(feature_pathlist, "w") as f:
            for idx in split_indices:
                f.write(f"feature-npy/{feature_files[idx].name}\n")

        with open(target_pathlist, "w") as f:
            for idx in split_indices:
                f.write(f"target-npy/{target_files[idx].name}\n")

        pathlist_files[split_name] = {
            "feature": feature_pathlist,
            "target": target_pathlist,
        }

    return pathlist_files


def ensure_test_data_exists(
    feature_dir: Path,
    target_dir: Path,
    num_samples: int = 300,
    train_count: int = 100,
    valid_count: int = 100,
    **kwargs,
) -> bool:
    """
    テストデータが存在しない場合は生成する

    Returns
    -------
        bool: データが新しく生成されたかどうか
    """
    # 既存のファイル数をチェック
    existing_features = (
        len(list(feature_dir.glob("*.npy"))) if feature_dir.exists() else 0
    )
    existing_targets = len(list(target_dir.glob("*.npy"))) if target_dir.exists() else 0

    if existing_features >= num_samples and existing_targets >= num_samples:
        print(
            f"Test data already exists (features: {existing_features}, targets: {existing_targets})"
        )
        return False

    print(f"Generating test data (need {num_samples} samples)...")
    generate_noisy_correlated_data(feature_dir, target_dir, num_samples, **kwargs)

    # pathlistファイルも生成
    base_dir = feature_dir.parent
    pathlist_files = create_pathlist_files(
        feature_dir=feature_dir,
        target_dir=target_dir,
        base_dir=base_dir,
        train_count=train_count,
        valid_count=valid_count,
        seed=kwargs.get("seed", 42),
    )

    print(f"Generated pathlist files: {list(pathlist_files.keys())}")
    return True


if __name__ == "__main__":
    # テスト用データの生成（300個）
    from pathlib import Path

    base_dir = Path(__file__).parent / "data" / "test_data"
    feature_dir = base_dir / "feature-npy"
    target_dir = base_dir / "target-npy"

    # 300個のノイジー相関データを生成
    generate_noisy_correlated_data(
        feature_dir=feature_dir,
        target_dir=target_dir,
        num_samples=300,
        feature_shape=(64,),
        num_classes=3,
        noise_level=0.3,
        seed=42,
    )

    # pathlistファイルも生成
    create_pathlist_files(
        feature_dir=feature_dir,
        target_dir=target_dir,
        base_dir=base_dir,
        train_count=100,
        valid_count=100,
        seed=42,
    )
