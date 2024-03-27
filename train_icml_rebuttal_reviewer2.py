import random
from typing import Optional

import fire
from ikarus.benchmark import benchmark
from ikarus.benchmark import IMTP
from ikarus.benchmark._benchmark import _mae_metrices
import numpy as np
import ring
from ring import ml
from ring.algorithms.generator import transforms

import wandb

dropout_rates = dict(
    seg2_1Seg=(0.0, 1.0),
    seg2_2Seg=(0.0, 1.0),
    seg3_2Seg=(0.0, 0.5),
    seg2_3Seg=(0.0, 1.0),
    seg3_3Seg=(0.9, 0.5),
    seg4_3Seg=(0.0, 0.5),
    seg5_4Seg=(0.0, 1.0),
    seg2_4Seg=(3 / 4, 1 / 4),
    seg3_4Seg=(3 / 4, 1 / 4),
    seg4_4Seg=(0.0, 1 / 4),
)


def output_transform_factory(link_names: list[str]):

    def _rename_links(d: dict[str, dict]):
        for key in list(d.keys()):
            if key in link_names:
                d[str(link_names.index(key))] = d.pop(key)
            else:
                assert key == "dt", f"{key} not in {link_names}"

    def output_transform(tree):
        X, y = tree

        for key in list(X.keys()):
            if key != "dt" and key not in link_names:
                X.pop(key)
                y.pop(key)

        B = X[link_names[0]]["gyr"].shape[0]

        draw = lambda p: np.random.binomial(1, p, size=B).astype(float)[:, None, None]
        fcs = {
            "seg3_3Seg": draw(1 - dropout_rates["seg3_3Seg"][1]),
            "seg4_3Seg": draw(1 - dropout_rates["seg4_3Seg"][1]),
        }

        for name in link_names:
            imu_rate, jointaxes_rate = dropout_rates[name]
            factor_imu = draw(1 - imu_rate)
            factor_ja = draw(1 - jointaxes_rate)

            if name in fcs:
                factor_ja = fcs[name]

            for gyraccmag in ["gyr", "acc", "mag"]:
                if gyraccmag in X[name]:
                    X[name][gyraccmag] *= factor_imu

            if "joint_axes" in X[name]:
                X[name]["joint_axes"] *= factor_ja

        _rename_links(X)
        _rename_links(y)
        return transforms._expand_then_flatten((X, y))

    return output_transform


def _make_ring(lam, warmstart: str | None, no_graph: bool):
    params = None if warmstart is None else f"~/params/0x{warmstart}.pickle"
    hidden_state_dim = 400 if ml.on_cluster() else 20
    message_dim = 200 if ml.on_cluster() else 10
    link_output_dim = 4
    link_output_normalize = True
    if no_graph:
        message_dim = 0
        link_output_dim = len(lam) * 4
        link_output_normalize = False
    ringnet = ml.RING(
        lam=lam,
        hidden_state_dim=hidden_state_dim,
        message_dim=message_dim,
        params=params,
        link_output_normalize=link_output_normalize,
        link_output_dim=link_output_dim,
    )
    if no_graph:
        ringnet = ml.base.NoGraph_FilterWrapper(ringnet, quat_normalize=True)
    ringnet = ml.base.ScaleX_FilterWrapper(ringnet)
    ringnet = ml.base.GroundTruthHeading_FilterWrapper(ringnet)
    return ringnet


def main(
    bs: int,
    episodes: int,
    # manual parse to list[str]; sepearator is ,
    tp: str,
    use_wandb: bool = False,
    wandb_project: str = "universal",
    wandb_name: str = None,
    warmstart: str = None,
    lr: float = 3e-3,
    seed: int = 1,
    kill_ep: Optional[int] = None,
    checkpoint: str = None,
    kill_after_hours: float = None,
):

    tp = tp.split(",")

    random.seed(seed)
    np.random.seed(seed)

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_name, config=locals())

    sys_noimu = (
        ring.io.load_example("exclude/standard_sys_rr_imp")
        .make_sys_noimu()[0]
        .delete_system([f"seg2_{i}Seg" for i in range(1, 4)])
    )

    ringnet = _make_ring(sys_noimu.link_parents, warmstart, no_graph=True)
    print(f"lam = {sys_noimu.link_parents}")

    callbacks, metrices_name = [], []

    def add_callback(
        imtp: IMTP, exp_id, motion_start, include_in_expval=True, twice=False
    ):
        cb = benchmark(
            imtp=imtp,
            exp_id=exp_id,
            motion_start=motion_start,
            filter=ringnet,
            return_cb=True,
        )
        callbacks.append(cb)
        if include_in_expval:
            for segment in imtp.segments:
                for _ in range((2 if twice else 1)):
                    metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    # 4SEG exp callbacks
    timings = {
        1: ["slow1", "fast"],
        2: ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    for exp_id in timings:
        for phase in timings[exp_id]:
            add_callback(
                IMTP(["seg5", "seg2", "seg3", "seg4"], joint_axes=True, sparse=True),
                exp_id,
                phase,
                twice=True,
            )

    # create one large "experimental validation" metric
    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    generator, (X_val, y_val) = ml.ml_utils.train_val_split(
        tp,
        bs,
        transform_gen=transforms.GeneratorTrafoLambda(
            output_transform_factory(sys_noimu.link_names)
        ),
    )

    callbacks += [
        ml.callbacks.EvalXyTrainingLoopCallback(
            ringnet,
            _mae_metrices,
            X_val,
            y_val,
            None,
            "val",
            link_names=sys_noimu.link_names,
        )
    ]

    optimizer = ml.make_optimizer(
        lr,
        episodes,
        n_steps_per_episode=6,
        skip_large_update_max_normsq=100.0,
    )

    ml.train_fn(
        generator,
        episodes,
        ringnet,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_after_seconds=(
            23.5 * 3600 if kill_after_hours is None else kill_after_hours * 3600
        ),
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=f"~/params/{ml.unique_id()}.pickle",
        callback_kill_after_episode=kill_ep,
        callback_save_params_track_metrices=[["exp_val_mae_deg"]],
        checkpoint=(
            None if checkpoint is None else f"~/.xxy_checkpoints/0x{checkpoint}.pickle"
        ),
        seed_network=seed,
        link_names=sys_noimu.link_names,
    )


if __name__ == "__main__":
    fire.Fire(main)
