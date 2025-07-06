from pathlib import Path

import numpy as np


def generate_multi_type_data(
    data_dir: Path,
    num_samples: int = 300,
    feature_shape: tuple[int, ...] = (64,),
    num_classes: int = 3,
    noise_level: float = 0.3,
    seed: int = 42,
) -> None:
    """
    複数タイプのテストデータを生成する（feature_vector, feature_variable, target_vector, target_scalar）

    Args:
        data_dir: データファイルを保存するベースディレクトリ
        num_samples: サンプル数
        feature_shape: 特徴量の形状
        num_classes: クラス数
        noise_level: ノイズの強さ (0.0-1.0)
        seed: 乱数シード
    """
    # 各データタイプのディレクトリを作成
    feature_vector_dir = data_dir / "feature_vector"
    feature_variable_dir = data_dir / "feature_variable"
    target_vector_dir = data_dir / "target_vector"
    target_scalar_dir = data_dir / "target_scalar"

    for dir_path in [
        feature_vector_dir,
        feature_variable_dir,
        target_vector_dir,
        target_scalar_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 乱数シードを設定
    np.random.seed(seed)

    # 各クラスに対応する基本パターンを生成
    class_patterns = []
    for class_idx in range(num_classes):
        pattern = np.random.randn(*feature_shape).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)
        class_patterns.append(pattern)

    # データを生成
    for i in range(num_samples):
        # ターゲットクラスをランダムに選択
        target_class = np.random.randint(0, num_classes, dtype=np.int64)

        # 基本パターンを取得
        base_pattern = class_patterns[target_class].copy()

        # 1. feature_vector: 固定長特徴ベクトル
        correlation_signal = base_pattern * (1.0 - noise_level)
        noise = np.random.randn(*feature_shape).astype(np.float32) * noise_level
        feature_vector = correlation_signal + noise
        feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)

        # 2. feature_variable: 可変長特徴データ
        variable_length = np.random.randint(5, 20)  # 5-19の可変長
        feature_variable = np.random.randn(variable_length, 32).astype(np.float32)

        # 3. target_vector: ベクトル形式のターゲット（従来のtarget）
        target_vector = target_class

        # 4. target_scalar: スカラー形式のターゲット（回帰値）
        target_scalar = float(target_class) + np.random.randn() * 0.1

        # ファイルに保存（各ディレクトリに同じファイル名で保存）
        sample_id = f"{i:03d}.npy"
        np.save(feature_vector_dir / sample_id, feature_vector)
        np.save(feature_variable_dir / sample_id, feature_variable)
        np.save(target_vector_dir / sample_id, target_vector)
        np.save(target_scalar_dir / sample_id, target_scalar)

    print(f"Generated {num_samples} multi-type test samples")
    print(f"Files saved to: {data_dir}")


def create_pathlist_files(
    data_dir: Path,
    base_dir: Path,
    train_count: int = 100,
    valid_count: int = 100,
    seed: int = 42,
) -> dict[str, Path]:
    """
    新しいマルチタイプデータ用のpathlistファイルを生成する

    Args:
        data_dir: データファイルのディレクトリ
        base_dir: pathlistファイルを保存するディレクトリ
        train_count: 訓練データ数
        valid_count: バリデーションデータ数
        seed: 乱数シード

    Returns
    -------
        dict: pathlistファイルのパス辞書
    """
    np.random.seed(seed)

    # feature_vectorディレクトリから利用可能なファイルのリストを取得
    feature_vector_dir = data_dir / "feature_vector"
    base_files = sorted(feature_vector_dir.glob("*.npy"))

    total_files = len(base_files)
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

        # pathlistファイルを作成（ファイル名のみを記録）
        with open(feature_pathlist, "w") as f:
            for idx in split_indices:
                file_name = base_files[idx].name  # XXX.npy
                f.write(f"{file_name}\n")

        with open(target_pathlist, "w") as f:
            for idx in split_indices:
                file_name = base_files[idx].name  # XXX.npy
                f.write(f"{file_name}\n")

        pathlist_files[split_name] = {
            "feature": feature_pathlist,
            "target": target_pathlist,
        }

    return pathlist_files


if __name__ == "__main__":
    base_dir = Path(__file__).parent / "data" / "test_data"
    generate_multi_type_data(base_dir, num_samples=300, seed=42)
    create_pathlist_files(base_dir, base_dir, train_count=200, valid_count=100, seed=42)
