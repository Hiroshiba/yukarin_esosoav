import numpy as np
from pathlib import Path
from typing import Tuple


def generate_test_data(
    feature_dir: Path,
    target_dir: Path,
    num_samples: int = 10,
    feature_shape: Tuple[int, ...] = (64,),
    num_classes: int = 3,
    seed: int = 42
) -> None:
    """
    テスト用のデータセットを生成する関数
    
    Args:
        feature_dir: 特徴量ファイルを保存するディレクトリ
        target_dir: ターゲットファイルを保存するディレクトリ
        num_samples: サンプル数
        feature_shape: 特徴量の形状
        num_classes: クラス数
        seed: 乱数シード
    """
    # ディレクトリを作成
    feature_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 乱数シードを設定
    np.random.seed(seed)
    
    # データを生成
    for i in range(num_samples):
        # 特徴量: 正規分布に従う乱数
        feature = np.random.randn(*feature_shape).astype(np.float32)
        
        # ターゲット: 0からnum_classes-1までのランダムな整数
        target = np.random.randint(0, num_classes, dtype=np.int64)
        
        # ファイルに保存
        feature_path = feature_dir / f"feature_{i:03d}.npy"
        target_path = target_dir / f"target_{i:03d}.npy"
        
        np.save(feature_path, feature)
        np.save(target_path, target)
    
    print(f"Generated {num_samples} test samples")
    print(f"Feature files saved to: {feature_dir}")
    print(f"Target files saved to: {target_dir}")


def ensure_test_data_exists(
    feature_dir: Path,
    target_dir: Path,
    num_samples: int = 10,
    **kwargs
) -> bool:
    """
    テストデータが存在しない場合は生成する
    
    Returns:
        bool: データが新しく生成されたかどうか
    """
    # 既存のファイル数をチェック
    existing_features = len(list(feature_dir.glob("*.npy"))) if feature_dir.exists() else 0
    existing_targets = len(list(target_dir.glob("*.npy"))) if target_dir.exists() else 0
    
    if existing_features >= num_samples and existing_targets >= num_samples:
        print(f"Test data already exists (features: {existing_features}, targets: {existing_targets})")
        return False
    
    print(f"Generating test data (need {num_samples} samples)...")
    generate_test_data(feature_dir, target_dir, num_samples, **kwargs)
    return True


if __name__ == "__main__":
    # テスト用データの生成
    from pathlib import Path
    
    base_dir = Path(__file__).parent / "data" / "test_data"
    feature_dir = base_dir / "feature-npy"
    target_dir = base_dir / "target-npy"
    
    generate_test_data(feature_dir, target_dir)