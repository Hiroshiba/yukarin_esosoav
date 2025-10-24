"""機械学習モデルの学習メインスクリプト"""

import argparse
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from hiho_pytorch_base.batch import BatchOutput, collate_dataset_output
from hiho_pytorch_base.config import Config
from hiho_pytorch_base.dataset import DatasetType, create_dataset, prefetch_datas
from hiho_pytorch_base.evaluator import (
    Evaluator,
    EvaluatorOutput,
    calculate_value,
)
from hiho_pytorch_base.generator import Generator
from hiho_pytorch_base.model import (
    DiscriminatorModelOutput,
    GeneratorModelOutput,
    Model,
)
from hiho_pytorch_base.network.discriminator import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from hiho_pytorch_base.network.predictor import Predictor, create_predictor
from hiho_pytorch_base.utility.pytorch_utility import (
    init_weights,
    make_optimizer,
    make_scheduler,
)
from hiho_pytorch_base.utility.train_utility import (
    DataNumProtocol,
    Logger,
    SaveManager,
    reduce_result,
)


def _delete_data_num(output: DataNumProtocol) -> dict[str, Any]:
    if not hasattr(output, "data_num"):
        raise ValueError("Output does not have 'data_num' attribute")
    return {k: v for k, v in asdict(output).items() if k != "data_num"}


@dataclass
class TrainingResults:
    """学習結果"""

    train_generator: GeneratorModelOutput
    train_discriminator: DiscriminatorModelOutput

    def to_summary_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書を生成"""
        return {
            DatasetType.TRAIN.value: {
                "generator": _delete_data_num(self.train_generator),
                "discriminator": _delete_data_num(self.train_discriminator),
            }
        }


@dataclass
class EvaluationResults:
    """評価結果"""

    test_generator: GeneratorModelOutput
    test_discriminator: DiscriminatorModelOutput
    eval: EvaluatorOutput | None
    valid: EvaluatorOutput | None

    def to_summary_dict(self) -> dict[str, Any]:
        """ログ出力用の辞書を生成"""
        summary = {
            DatasetType.TEST.value: {
                "generator": _delete_data_num(self.test_generator),
                "discriminator": _delete_data_num(self.test_discriminator),
            }
        }
        if self.eval is not None:
            summary[DatasetType.EVAL.value] = _delete_data_num(self.eval)
        if self.valid is not None:
            summary[DatasetType.VALID.value] = _delete_data_num(self.valid)
        return summary


@dataclass
class TrainingContext:
    """学習に必要な全てのオブジェクトをまとめる"""

    config: Config
    train_loader: DataLoader
    test_loader: DataLoader
    eval_loader: DataLoader | None
    valid_loader: DataLoader | None
    model: Model
    predictor: Predictor
    evaluator: Evaluator
    generator_optimizer: torch.optim.Optimizer
    discriminator_optimizer: torch.optim.Optimizer
    generator_scaler: GradScaler
    discriminator_scaler: GradScaler
    logger: Logger
    generator_scheduler: torch.optim.lr_scheduler.LRScheduler | None
    discriminator_scheduler: torch.optim.lr_scheduler.LRScheduler | None
    save_manager: SaveManager
    device: str
    epoch: int
    iteration: int
    snapshot_path: Path


class FirstEpochOrderedSampler(Sampler[int]):
    """初回エポックは指定順序、以降はランダムサンプリング。prefetchに有効。"""

    def __init__(self, first_indices: list[int]) -> None:
        self.first_indices = first_indices
        self.first_epoch = True

    def __iter__(self) -> Iterator[int]:  # noqa: D105
        if self.first_epoch:
            self.first_epoch = False
            return iter(self.first_indices)
        else:
            indices_tensor = torch.tensor(self.first_indices)
            return iter(indices_tensor[torch.randperm(len(indices_tensor))].tolist())

    def __len__(self) -> int:  # noqa: D105
        return len(self.first_indices)


def create_data_loader(
    config: Config,
    dataset: Dataset,
    for_train: bool,
    for_eval: bool,
    first_indices: list[int] | None,
) -> DataLoader:
    """DataLoaderを作成"""
    batch_size = config.train.eval_batch_size if for_eval else config.train.batch_size

    if first_indices is not None:
        sampler = FirstEpochOrderedSampler(first_indices)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.train.preprocess_workers,
        collate_fn=collate_dataset_output,
        pin_memory=config.train.use_gpu,
        drop_last=for_train,
        timeout=0 if config.train.preprocess_workers == 0 else 30,
        persistent_workers=config.train.preprocess_workers > 0,
    )


def setup_training_context(config_yaml_path: Path, output_dir: Path) -> TrainingContext:
    """TrainingContextを作成"""
    # config
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    config.add_git_info()
    config.validate_config()

    # dataset
    datasets = create_dataset(config.dataset)

    # prefetch
    train_indices = torch.randperm(len(datasets.train)).tolist()
    datas = datasets.train.datas + datasets.test.datas
    datas += datasets.eval.datas if datasets.eval is not None else []
    datas += datasets.valid.datas if datasets.valid is not None else []
    threading.Thread(
        target=prefetch_datas, args=(datas, config.train.prefetch_workers), daemon=True
    ).start()

    # data loader
    train_loader = create_data_loader(
        config,
        datasets.train,
        for_train=True,
        for_eval=False,
        first_indices=train_indices,
    )
    test_loader = create_data_loader(
        config, datasets.test, for_train=False, for_eval=False, first_indices=None
    )
    eval_loader = (
        create_data_loader(
            config, datasets.eval, for_train=False, for_eval=True, first_indices=None
        )
        if datasets.eval is not None
        else None
    )
    valid_loader = (
        create_data_loader(
            config, datasets.valid, for_train=False, for_eval=True, first_indices=None
        )
        if datasets.valid is not None
        else None
    )

    # predictor
    predictor = create_predictor(config.network)
    device = "cuda" if config.train.use_gpu else "cpu"
    if config.train.pretrained_predictor_path is not None:
        state_dict = torch.load(
            config.train.pretrained_predictor_path, map_location=device
        )
        predictor.load_state_dict(state_dict)
    if config.train.pretrained_vocoder_path is not None:
        state_dict = torch.load(
            config.train.pretrained_vocoder_path, map_location=device
        )
        predictor.vocoder.load_state_dict(state_dict)
    print("predictor:", predictor)

    # model
    # predictor_scripted: Predictor = torch.compile(predictor)  # type: ignore
    predictor_scripted = predictor  # FIXME: コンパイルは一旦保留
    mpd = MultiPeriodDiscriminator(
        initial_channel=config.network.discriminator.mpd_initial_channel
    )
    msd = MultiScaleDiscriminator(
        initial_channel=config.network.discriminator.msd_initial_channel
    )
    model = Model(
        model_config=config.model,
        predictor=predictor_scripted,
        mpd=mpd,
        msd=msd,
    )
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)
    model.to(device)

    # evaluator
    generator = Generator(
        config=config, predictor=predictor_scripted, use_gpu=config.train.use_gpu
    )
    evaluator = Evaluator(generator=generator)

    # optimizer
    generator_optimizer = make_optimizer(
        config_dict=config.train.generator_optimizer, model=predictor
    )
    discriminator_optimizer = make_optimizer(
        config_dict=config.train.discriminator_optimizer,
        model=nn.ModuleList([mpd, msd]),
    )
    generator_scaler = GradScaler(device, enabled=config.train.use_amp)
    discriminator_scaler = GradScaler(device, enabled=config.train.use_amp)

    # logger
    logger = Logger(
        config_dict=config_dict,
        project_category=config.project.category,
        project_name=config.project.name,
        output_dir=output_dir,
    )

    # scheduler
    generator_scheduler = (
        make_scheduler(
            config_dict=config.train.generator_scheduler,
            optimizer=generator_optimizer,
        )
        if config.train.generator_scheduler is not None
        else None
    )
    discriminator_scheduler = (
        make_scheduler(
            config_dict=config.train.discriminator_scheduler,
            optimizer=discriminator_optimizer,
        )
        if config.train.discriminator_scheduler is not None
        else None
    )

    # save
    save_manager = SaveManager(
        predictor=predictor,
        prefix="predictor_",
        output_dir=output_dir,
        top_num=config.train.model_save_num,
        last_num=config.train.model_save_num,
    )

    return TrainingContext(
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        eval_loader=eval_loader,
        valid_loader=valid_loader,
        model=model,
        predictor=predictor,
        evaluator=evaluator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scaler=generator_scaler,
        discriminator_scaler=discriminator_scaler,
        logger=logger,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        save_manager=save_manager,
        device=device,
        epoch=0,
        iteration=0,
        snapshot_path=output_dir / "snapshot.pth",
    )


def load_snapshot(context: TrainingContext) -> None:
    """学習状態を復元"""
    snapshot = torch.load(context.snapshot_path, map_location=context.device)

    context.model.load_state_dict(snapshot["model"])
    context.generator_optimizer.load_state_dict(snapshot["generator_optimizer"])
    context.discriminator_optimizer.load_state_dict(snapshot["discriminator_optimizer"])
    context.generator_scaler.load_state_dict(snapshot["generator_scaler"])
    context.discriminator_scaler.load_state_dict(snapshot["discriminator_scaler"])
    context.logger.load_state_dict(snapshot["logger"])

    context.iteration = snapshot["iteration"]
    context.epoch = snapshot["epoch"]

    if context.generator_scheduler is not None:
        context.generator_scheduler.last_epoch = context.epoch
    if context.discriminator_scheduler is not None:
        context.discriminator_scheduler.last_epoch = context.epoch


def train_one_epoch(context: TrainingContext) -> TrainingResults:
    """１エポックの学習処理"""
    context.model.train()
    if hasattr(context.generator_optimizer, "train"):
        context.generator_optimizer.train()  # type: ignore
    if hasattr(context.discriminator_optimizer, "train"):
        context.discriminator_optimizer.train()  # type: ignore

    gradient_accumulation = context.config.train.gradient_accumulation
    context.generator_optimizer.zero_grad()  # NOTE: 端数分の勾配を消す
    context.discriminator_optimizer.zero_grad()

    generator_results: list[GeneratorModelOutput] = []
    discriminator_results: list[DiscriminatorModelOutput] = []

    for batch_index, batch in enumerate(context.train_loader, start=1):
        batch = batch.to_device(context.device, non_blocking=True)

        # Discriminatorの更新
        with autocast(context.device, enabled=context.config.train.use_amp):
            spec1_list, spec2_list, pred_wave_list = context.model.forward(batch)
            discriminator_output = context.model.calc_discriminator(
                batch=batch,
                pred_wave_list=pred_wave_list,
            )
        discriminator_loss = discriminator_output.loss / gradient_accumulation
        if discriminator_loss.isnan():
            raise ValueError("discriminator loss is NaN")

        context.discriminator_scaler.scale(discriminator_loss).backward()

        if batch_index % gradient_accumulation == 0:
            context.discriminator_scaler.step(context.discriminator_optimizer)
            context.discriminator_scaler.update()
            context.discriminator_optimizer.zero_grad()

        discriminator_results.append(discriminator_output.detach_cpu())

        # Generatorの更新
        with autocast(context.device, enabled=context.config.train.use_amp):
            generator_output = context.model.calc_generator(
                batch=batch,
                spec1_list=spec1_list,
                spec2_list=spec2_list,
                pred_wave_list=pred_wave_list,
            )
        generator_loss = generator_output.loss / gradient_accumulation
        if generator_loss.isnan():
            raise ValueError("generator loss is NaN")

        context.generator_scaler.scale(generator_loss).backward()

        if batch_index % gradient_accumulation == 0:
            context.generator_scaler.step(context.generator_optimizer)
            context.generator_scaler.update()
            context.generator_optimizer.zero_grad()
            context.iteration += 1

        generator_results.append(generator_output.detach_cpu())

    if context.generator_scheduler is not None:
        context.generator_scheduler.step()
    if context.discriminator_scheduler is not None:
        context.discriminator_scheduler.step()

    return TrainingResults(
        train_generator=reduce_result(generator_results),
        train_discriminator=reduce_result(discriminator_results),
    )


@torch.no_grad()
def evaluate(context: TrainingContext) -> EvaluationResults:
    """評価値を計算する"""
    context.model.eval()
    if hasattr(context.generator_optimizer, "eval"):
        context.generator_optimizer.eval()  # type: ignore
    if hasattr(context.discriminator_optimizer, "eval"):
        context.discriminator_optimizer.eval()  # type: ignore

    batch: BatchOutput

    # test評価
    test_generator_list: list[GeneratorModelOutput] = []
    test_discriminator_list: list[DiscriminatorModelOutput] = []
    for batch in context.test_loader:
        batch = batch.to_device(context.device, non_blocking=True)
        spec1_list, spec2_list, pred_wave_list = context.model.forward(batch)
        generator_result = context.model.calc_generator(
            batch=batch,
            spec1_list=spec1_list,
            spec2_list=spec2_list,
            pred_wave_list=pred_wave_list,
        )
        discriminator_result = context.model.calc_discriminator(
            batch=batch,
            pred_wave_list=pred_wave_list,
        )
        test_generator_list.append(generator_result.detach_cpu())
        test_discriminator_list.append(discriminator_result.detach_cpu())
    test_generator = reduce_result(test_generator_list)
    test_discriminator = reduce_result(test_discriminator_list)

    # eval評価
    eval_result = None
    if context.eval_loader is not None:
        eval_result_list: list[EvaluatorOutput] = []
        for batch in context.eval_loader:
            batch = batch.to_device(context.device, non_blocking=True)
            evaluator_result: EvaluatorOutput = context.evaluator(batch)
            eval_result_list.append(evaluator_result.detach_cpu())
        eval_result = reduce_result(eval_result_list)

    # valid評価
    valid_result = None
    if context.valid_loader is not None:
        valid_result_list: list[EvaluatorOutput] = []
        for batch in context.valid_loader:
            batch = batch.to_device(context.device, non_blocking=True)
            evaluator_result: EvaluatorOutput = context.evaluator(batch)
            valid_result_list.append(evaluator_result.detach_cpu())
        valid_result = reduce_result(valid_result_list)

    return EvaluationResults(
        test_generator=test_generator,
        test_discriminator=test_discriminator,
        eval=eval_result,
        valid=valid_result,
    )


def save_predictor(
    context: TrainingContext, evaluation_results: EvaluationResults
) -> None:
    """評価結果に基づいてPredictorを保存する"""
    if evaluation_results.valid is not None:
        evaluation_value = float(calculate_value(evaluation_results.valid))
    else:
        evaluation_value = 0

    context.save_manager.save(value=evaluation_value, step=context.epoch)


def save_checkpoint(context: TrainingContext) -> None:
    """チェックポイント保存する"""
    torch.save(
        {
            "model": context.model.state_dict(),
            "generator_optimizer": context.generator_optimizer.state_dict(),
            "discriminator_optimizer": context.discriminator_optimizer.state_dict(),
            "generator_scaler": context.generator_scaler.state_dict(),
            "discriminator_scaler": context.discriminator_scaler.state_dict(),
            "logger": context.logger.state_dict(),
            "iteration": context.iteration,
            "epoch": context.epoch,
        },
        context.snapshot_path,
    )


def should_log_epoch(context: TrainingContext) -> bool:
    """ログ出力するかどうか判定する"""
    return context.epoch % context.config.train.log_epoch == 0


def should_eval_epoch(context: TrainingContext) -> bool:
    """評価実行するかどうか判定する"""
    return context.epoch % context.config.train.eval_epoch == 0


def should_snapshot_epoch(context: TrainingContext) -> bool:
    """スナップショット保存するかどうか判定する"""
    return context.epoch % context.config.train.snapshot_epoch == 0


def training_loop(context: TrainingContext) -> None:
    """学習ループ"""
    for _ in range(context.config.train.stop_epoch):
        context.epoch += 1
        if context.epoch > context.config.train.stop_epoch:
            break

        training_results = train_one_epoch(context)

        if should_log_epoch(context):
            summary = {
                "iteration": context.iteration,
                "epoch": context.epoch,
                "lr": context.generator_optimizer.param_groups[0]["lr"],
                "lr_discriminator": context.discriminator_optimizer.param_groups[0][
                    "lr"
                ],
            }
            summary.update(training_results.to_summary_dict())

            if should_eval_epoch(context):
                evaluation_results = evaluate(context)
                summary.update(evaluation_results.to_summary_dict())
                save_predictor(context, evaluation_results)

            context.logger.log(summary=summary, step=context.epoch)

        if should_snapshot_epoch(context):
            save_checkpoint(context)


def train(config_yaml_path: Path, output_dir: Path) -> None:
    """機械学習モデルを学習する。スナップショットがあれば再開する。"""
    context = setup_training_context(config_yaml_path, output_dir)

    if context.snapshot_path.exists():
        load_snapshot(context)

    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(context.config.to_dict(), f)

    try:
        training_loop(context)
    finally:
        context.logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    train(**vars(parser.parse_args()))
