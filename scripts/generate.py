"""学習済みモデルからの推論・生成スクリプト"""

import argparse
import re
from pathlib import Path

import yaml
from tqdm import tqdm

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import create_dataset
from hiho_pytorch_base.generator import Generator
from utility.save_arguments import save_arguments


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + f"{iteration}.pth")
        assert model_path.exists()
    return model_path


def generate(
    model_dir: Path,
    model_iteration: int | None,
    model_config: Path | None,
    output_dir: Path,
    use_gpu: bool,
):
    """学習済みモデルを使用して推論を実行"""
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    dataset = create_dataset(config.dataset).test
    for data in tqdm(dataset, desc="generate"):
        feature_vector = data.feature_vector
        feature_variable = data.feature_variable
        target = data.target_vector
        output = generator.forward(feature_vector, [feature_variable])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--use_gpu", action="store_true")
    generate(**vars(parser.parse_args()))
