import random
from typing import Optional

import fire
import numpy as np
import x_xy
from x_xy import exp
from x_xy import ml
from x_xy.algorithms.generator import transforms

import wandb

dropout_rates = dict(
    seg2_1Seg=(0.0, 1.0),
    seg2_2Seg=(0.0, 1.0),
    seg3_2Seg=(0.0, 0.5),
    seg2_3Seg=(0.0, 1.0),
    seg3_3Seg=(2 / 3, 0.5),
    seg4_3Seg=(0.0, 0.5),
    seg5_4Seg=(0.0, 1.0),
    seg2_4Seg=(3 / 4, 1 / 4),
    seg3_4Seg=(3 / 4, 1 / 4),
    seg4_4Seg=(0.0, 1 / 4),
)


def output_transform_factory(link_names):

    def _rename_links(d: dict[str, dict]):
        for key in list(d.keys()):
            if key in link_names:
                d[str(link_names.index(key))] = d.pop(key)

    def output_transform(tree):
        X, y = tree
        segments = list(set(X.keys()) - set(["dt"]))

        any_segment = X[segments[0]]
        assert any_segment["gyr"].ndim == 3, f"{any_segment['gyr'].shape}"
        B = any_segment["gyr"].shape[0]

        draw = lambda p: np.random.binomial(1, p, size=B).astype(float)[:, None, None]
        fcs = {
            "seg3_3Seg": draw(1 - dropout_rates["seg3_3Seg"][1]),
            "seg4_3Seg": draw(1 - dropout_rates["seg4_3Seg"][1]),
        }

        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            factor_imu = draw(1 - imu_rate)
            factor_ja = draw(1 - jointaxes_rate)

            if segments == "seg3_3Seg":
                factor_imu = 0.0

            if segments in fcs:
                factor_ja = fcs[segments]

            for gyraccmag in ["gyr", "acc", "mag"]:
                if gyraccmag in X[segments]:
                    X[segments][gyraccmag] *= factor_imu

            if "joint_axes" in X[segments]:
                X[segments]["joint_axes"] *= factor_ja

        _rename_links(X)
        _rename_links(y)
        return transforms._expand_then_flatten((X, y))

    return output_transform


def _make_ring(lam, warmstart: str | None):
    params = None if warmstart is None else f"~/params/0x{warmstart}.pickle"
    hidden_state_dim = 400 if ml.on_cluster() else 20
    message_dim = 200 if ml.on_cluster() else 10
    ring = ml.RING(
        lam=lam,
        hidden_state_dim=hidden_state_dim,
        message_dim=message_dim,
        params=params,
    )
    ring = ml.base.ScaleX_FilterWrapper(ring)
    ring = ml.base.GroundTruthHeading_FilterWrapper(ring)
    return ring


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

    sys_noimu = x_xy.io.load_example("exclude/standard_sys_rr_imp").make_sys_noimu()[0]
    ring = _make_ring(sys_noimu.link_parents, warmstart)
    print(f"lam = {sys_noimu.link_parents}")

    callbacks, metrices_name = [], []

    def add_callback(
        imtp: exp.IMTP, exp_id, motion_start, include_in_expval=True, twice=False
    ):
        cb = exp.benchmark_fn(ring, imtp, exp_id, motion_start, return_cb=True)
        callbacks.append(cb)
        if include_in_expval:
            for segment in imtp.segments:
                for _ in range((2 if twice else 1)):
                    metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    # 1SEG exp callbacks
    timings = {
        "S_07": ["slow_fast_mix", "slow_fast_freeze_mix"],
    }

    for anchor_1Seg in ["seg1", "seg5", "seg2", "seg3", "seg4"]:
        for exp_id in timings:
            for phase in timings[exp_id]:
                add_callback(
                    exp.IMTP([anchor_1Seg], model_name_suffix=f"_{anchor_1Seg}"),
                    exp_id,
                    phase,
                )

    # 4SEG exp callbacks
    timings = {
        "S_06": ["slow1", "fast"],
        "S_07": ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    for exp_id in timings:
        for phase in timings[exp_id]:
            add_callback(
                exp.IMTP(
                    ["seg5", "seg2", "seg3", "seg4"], joint_axes=True, sparse=True
                ),
                exp_id,
                phase,
                twice=True,
            )

    # 2 Seg with flexible IMUs callbacks
    axes_S_06_07 = {
        "xaxis": ("seg5", "seg2"),
        "yaxis": ("seg2", "seg3"),
        "zaxis": ("seg3", "seg4"),
    }
    axes = {
        "S_06": axes_S_06_07,
        "S_07": axes_S_06_07,
        "S_16": {"left": ("seg1", "seg4"), "right": ("seg2", "seg3")},
    }

    timings.update(dict(S_16=["gait_slow", "gait_fast"]))
    for exp_id in timings:
        for phase in timings[exp_id]:
            for axis in axes[exp_id]:
                add_callback(
                    exp.IMTP(
                        list(axes[exp_id][axis]),
                        flex=True,
                        joint_axes=True,
                        model_name_suffix="_" + axis,
                    ),
                    exp_id,
                    phase,
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
            ring,
            exp.benchmark._mae_metrices,
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
        ring,
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
    )


if __name__ == "__main__":
    fire.Fire(main)
