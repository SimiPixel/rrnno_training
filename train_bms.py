import random
from typing import Optional

import fire
import jax
import numpy as np
import x_xy
from x_xy.algorithms.generator import transforms
from x_xy.subpkgs import ml
from x_xy.subpkgs import sys_composer

import wandb

natural_units_X_trafo = ml.convenient.rescale_natural_units_X_transform

dropout_rates = dict(
    seg2_2Seg=(0.0, 1.0),
    seg3_2Seg=(0.0, 0.9),
    seg2_3Seg=(0.0, 1.0),
    seg3_3Seg=(1.0, 0.9),
    seg4_3Seg=(0.0, 0.9),
)


def output_transform_factory(factor_ja_overwrite: float | None):
    def output_transform(tree):
        X, y = tree
        X = x_xy.utils.pytree_deepcopy(X)
        dt = X.pop("dt", None)

        any_segment = X[list(X.keys())[0]]
        assert any_segment["gyr"].ndim == 3, f"{any_segment['gyr'].shape}"
        bs, N, _ = any_segment["gyr"].shape

        if dt is not None:
            dt = np.repeat(dt[:, None, :], N, axis=1)

        draw = lambda p: np.random.binomial(1, p, size=bs).astype(float)[:, None, None]
        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            if segments not in X:
                continue

            factor_imu = draw(1 - imu_rate)
            factor_ja = draw(1 - jointaxes_rate)

            if factor_ja_overwrite is not None:
                factor_ja = factor_ja_overwrite

            for gyraccmag in ["gyr", "acc", "mag"]:
                if gyraccmag in X[segments]:
                    X[segments][gyraccmag] *= factor_imu

            if "joint_axes" in X[segments]:
                X[segments]["joint_axes"] *= factor_ja

            if dt is not None:
                X[segments]["dt"] = dt

        return natural_units_X_trafo(X), y

    return output_transform


def rnno_fn_factory(rnno: bool):
    def rnno_fn(sys):
        if rnno:
            sys = None
        return ml.make_rnno(
            sys,
            400 if ml.on_cluster() else 25,
            200 if ml.on_cluster() else 10,
            stack_rnn_cells=2,
            layernorm=True,
            keep_toRoot_output=True,
        )

    return rnno_fn


def main(
    bs: int,
    episodes: int,
    # manual parse to list[str]; sepearator is ,
    tp: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "universal",
    wandb_name: str = None,
    warmstart: str = None,
    lr: float = 3e-3,
    seed: int = 1,
    kill_ep: Optional[int] = None,
    checkpoint: str = None,
    kill_after_hours: float = None,
    rand_sampling_rates: bool = False,
    rnno: bool = False,
    known_ja: bool = False,
):
    assert tp is not None

    if tp is not None:
        tp = tp.split(",")

    random.seed(seed)
    np.random.seed(seed)

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_name, config=locals())

    rnno_fn = rnno_fn_factory(rnno)

    callbacks, metrices_name = [], []

    def add_callback(cb, sys, include_in_expval=True, twice=False):
        callbacks.append(cb)
        if include_in_expval:
            for segment in sys.findall_segments():
                for _ in range((2 if twice else 1)):
                    metrices_name.append([cb.metric_identifier, "mae_deg", segment])

    # 3SEG exp callbacks
    sys = ml.convenient.load_1Seg2Seg3Seg4Seg_system(
        anchor_3Seg="seg2", delete_inner_imus=True
    )
    exp_id = "S_07"
    timings = {
        "S_06": ["slow1", "fast"],
        "S_07": ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    for phase in timings[exp_id]:
        for flex in [False, True]:
            for ja in [False, True]:
                cb = ml.convenient.build_experimental_validation_callback2(
                    rnno_fn,
                    sys,
                    exp_id,
                    phase,
                    jointaxes=ja,
                    rootincl=True,
                    flex=flex,
                    X_transform=natural_units_X_trafo,
                    dt=rand_sampling_rates,
                )
                if rnno:
                    if not flex:
                        if known_ja and ja:
                            add_callback(cb, sys)
                        else:
                            add_callback(cb, sys)
                else:
                    add_callback(cb, sys)

    # create one large "experimental validation" metric
    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    sys = ml.convenient.load_1Seg2Seg3Seg4Seg_system(
        anchor_3Seg="seg2", add_suffix_to_linknames=True
    )
    sys_noimu = sys_composer.make_sys_noimu(sys)[0]
    del sys
    network = rnno_fn(sys_noimu)
    del sys_noimu

    generator, (X_val, y_val) = ml.convenient.train_val_split(
        tp,
        bs,
        transform_gen=transforms.GeneratorTrafoLambda(
            output_transform_factory(1.0 if known_ja else None)
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
    )

    ml.train(
        generator,
        episodes,
        network,
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
        initial_params=None if warmstart is None else f"~/params/0x{warmstart}.pickle",
        key_network=jax.random.PRNGKey(seed),
        checkpoint=(
            None if checkpoint is None else f"~/.xxy_checkpoints/0x{checkpoint}.pickle"
        ),
    )


if __name__ == "__main__":
    fire.Fire(main)
