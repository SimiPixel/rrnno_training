import random
from typing import Optional

import fire
import jax
import numpy as np
import wandb
import x_xy
from x_xy.algorithms.generator import transforms
from x_xy.subpkgs import ml
from x_xy.subpkgs import sys_composer

# 2Seg: seg2 - seg3
# 3Seg: seg2 - seg3 - seg4
# 4Seg: seg5 - seg2 - seg3 - seg4
dropout_rates1 = dict(
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
dropout_rates2 = dict(
    seg2_2Seg=(0.0, 1.0),
    seg3_2Seg=(0.0, 0.0),
    seg2_3Seg=(0.0, 1.0),
    seg3_3Seg=(1.0, 0.0),
    seg4_3Seg=(0.0, 0.0),
    seg5_4Seg=(0.0, 1.0),
    seg2_4Seg=(1.0, 0.0),
    seg3_4Seg=(1.0, 0.0),
    seg4_4Seg=(0.0, 0.0),
)
dropout_rates3 = dict(
    seg2_2Seg=(0.0, 1.0),
    seg3_2Seg=(0.0, 2 / 3),
    seg2_3Seg=(0.0, 1.0),
    seg3_3Seg=(2 / 3, 2 / 3),
    seg4_3Seg=(0.0, 2 / 3),
    seg5_4Seg=(0.0, 1.0),
    seg2_4Seg=(3 / 4, 0.0),
    seg3_4Seg=(3 / 4, 0.0),
    seg4_4Seg=(0.0, 0.0),
)
dropout_configs = {1: dropout_rates1, 2: dropout_rates2, 3: dropout_rates3}


natural_units_X_trafo = ml.convenient.rescale_natural_units_X_transform


def output_transform_factory(dropout_rates, joint_axes_aug: bool):
    def output_transform(tree):
        X, y = tree
        any_segment = X[list(X.keys())[0]]
        assert any_segment["gyr"].ndim == 3, f"{any_segment['gyr'].shape}"
        bs = any_segment["gyr"].shape[0]

        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            draw = lambda p: np.random.binomial(1, p, size=bs).astype(float)[
                :, None, None
            ]
            factor_imu = draw(1 - imu_rate)
            factor_ja = draw(1 - jointaxes_rate)

            for gyraccmag in ["gyr", "acc", "mag"]:
                if gyraccmag in X[segments]:
                    X[segments][gyraccmag] *= factor_imu

            if "joint_axes" in X[segments]:
                X[segments]["joint_axes"] *= factor_ja

                if joint_axes_aug:
                    X[segments]["joint_axes"] *= np.random.choice([-1.0, 1.0], size=bs)[
                        :, None, None
                    ]

        return natural_units_X_trafo(X), y

    return output_transform


def _make_rnno_fn(
    hidden_state_dim: int,
    stack_rnn_cells: int,
    layernorm: bool,
    stop_grads: bool,
    lstm: bool,
):
    def rnno_fn(sys):
        return ml.make_rnno(
            sys,
            hidden_state_dim if ml.on_cluster() else 25,
            200 if ml.on_cluster() else 10,
            stack_rnn_cells=stack_rnn_cells,
            layernorm=layernorm,
            send_message_stop_grads=stop_grads,
            keep_toRoot_output=True,
            cell_type="lstm" if lstm else "gru",
        )

    return rnno_fn


def main(
    bs: int,
    episodes: int,
    # manual parse to list[str]; sepearator is ,
    tp: Optional[str] = None,
    hidden_state_dim: int = 400,
    use_wandb: bool = False,
    wandb_project: str = "universal",
    wandb_name: str = None,
    warmstart: str = None,
    lr: float = 3e-3,
    layernorm: bool = False,
    stack_rnn: int = 1,
    lstm: bool = False,
    dropout_config: int = 1,
    seed: int = 1,
    kill_ep: Optional[int] = None,
    ja_aug: bool = False,
):
    assert tp is not None

    if tp is not None:
        tp = tp.split(",")

    random.seed(seed)
    np.random.seed(seed)

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_name, config=locals())

    two_days = False
    if episodes > 5000:
        two_days = True

    rnno_fn = _make_rnno_fn(
        hidden_state_dim, stack_rnn, layernorm, stop_grads=False, lstm=lstm
    )

    callbacks, metrices_name = [], []

    def add_callback(cb, sys, include_in_expval=True, twice=False):
        callbacks.append(cb)
        if include_in_expval:
            for segment in sys.findall_segments():
                for _ in range((2 if twice else 1)):
                    metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    # standard experimental callbacks
    timings = {
        "S_06": ["slow1", "fast"],
        "S_07": ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    for exp_id in timings:
        for phase in timings[exp_id]:
            for sys, ja in zip(
                [
                    ml.convenient.load_2Seg3Seg4Seg_system(
                        anchor_3Seg="seg2", delete_inner_imus=True
                    ),
                    ml.convenient.load_2Seg3Seg4Seg_system(
                        anchor_4Seg="seg5", delete_inner_imus=True
                    ),
                ],
                [False, True],
            ):
                cb = ml.convenient.build_experimental_validation_callback2(
                    rnno_fn,
                    sys,
                    exp_id,
                    phase,
                    jointaxes=ja,
                    rootincl=True,
                    X_transform=natural_units_X_trafo,
                )
                add_callback(cb, sys)

    # 2 Seg with flexible IMUs callbacks
    axes = {
        "xaxis": ("seg5", "seg2"),
        "yaxis": ("seg2", "seg3"),
        "zaxis": ("seg3", "seg4"),
    }

    def load_sys_flexible(fem, tib, suffix):
        return (
            x_xy.load_example("knee_flexible_imus")
            .change_model_name(suffix="_" + suffix)
            .change_link_name("femur", fem)
            .change_link_name("tibia", tib)
        )

    for phase in timings:
        for axis in axes:
            sys = load_sys_flexible(*axes[axis], suffix=axis)
            cb = ml.convenient.build_experimental_validation_callback2(
                rnno_fn,
                sys,
                "S_07",
                phase,
                jointaxes=True,
                flex=True,
                rootincl=True,
                X_transform=natural_units_X_trafo,
            )
            add_callback(cb, sys)

    del sys

    # create one large "experimental validation" metric
    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    sys = ml.convenient.load_2Seg3Seg4Seg_system(
        "seg2", "seg2", "seg5", add_suffix_to_linknames=True
    )
    sys_noimu = sys_composer.make_sys_noimu(sys)[0]
    del sys
    network = rnno_fn(sys_noimu)
    del sys_noimu

    generator, (X_val, y_val) = ml.convenient.train_val_split(
        tp,
        bs,
        transform_gen=transforms.GeneratorTrafoLambda(
            output_transform_factory(dropout_configs[dropout_config], ja_aug)
        ),
    )

    callbacks += [
        ml.EvalXyTrainingLoopCallback(
            network, ml.convenient._mae_metrices, X_val, y_val, "val"
        )
    ]

    optimizer = ml.make_optimizer(
        lr,
        episodes,
        n_steps_per_episode=6,
        skip_large_update_max_normsq=100.0,
        cos_decay_twice=two_days,
    )

    callback_kill_after_seconds = 23.75 * 3600
    if two_days:
        callback_kill_after_seconds *= 2

    ml.train(
        generator,
        episodes,
        network,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_after_seconds=callback_kill_after_seconds,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=f"~/params/{ml.unique_id()}.pickle",
        callback_kill_after_episode=kill_ep,
        callback_save_params_track_metrices=[["exp_val_mae_deg"]],
        initial_params=None if warmstart is None else f"~/params/0x{warmstart}.pickle",
        key_network=jax.random.PRNGKey(seed),
    )


if __name__ == "__main__":
    fire.Fire(main)
