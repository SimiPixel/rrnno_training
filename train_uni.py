import random
from typing import Optional

import fire
import jax
import numpy as np
from tree_utils._tree_utils import PyTree
import wandb
import x_xy
from x_xy.algorithms.generator import transforms
from x_xy.subpkgs import benchmark
from x_xy.subpkgs import ml
from x_xy.subpkgs import sys_composer
from x_xy.subpkgs.ml.ml_utils import Logger

natural_units_X_trafo = ml.convenient.rescale_natural_units_X_transform


class SaddleCallback(ml.callbacks.TrainingLoopCallback):
    def __init__(self, rnno_fn, ja_i: list[float], ja_o) -> None:
        filter = ml.InitApplyFnFilter(rnno_fn, X_transform=natural_units_X_trafo)
        self.predict = jax.jit(
            benchmark.saddle(
                filter,
                "S_13",
                "slow_fast_freeze_mix",
                None,
                "right",
                warmup=1000,
                factory=True,
                ja_inner=ja_i,
                ja_outer=ja_o,
            )
        )

        def to_str(eles: list[float]) -> str:
            s = ""
            for ele in eles:
                s += str(int(ele))
            return s

        self.identifier = f"saddle_i_{to_str(ja_i)}_o_{to_str(ja_o)}"

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
        opt_state: PyTree,
    ) -> None:
        if (i_episode % 5) == 0:
            self.last_metric = {self.identifier: self.predict(params)}
        metrices.update(self.last_metric)


# 2Seg: seg2 - seg3
# 3Seg: seg2 - seg3 - seg4
# 4Seg: seg5 - seg2 - seg3 - seg4
dropout_rates1 = dict(
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


def output_transform_factory(dropout_rates, joint_axes_aug: bool):
    def output_transform(tree):
        X, y = tree
        X = X.copy()
        two_dof = X.pop("2DOF", None)
        X.pop("pos", None)
        any_segment = X[list(X.keys())[0]]
        assert any_segment["gyr"].ndim == 3, f"{any_segment['gyr'].shape}"
        bs = any_segment["gyr"].shape[0]

        draw = lambda p: np.random.binomial(1, p, size=bs).astype(float)[:, None, None]
        fcs = {
            "seg3_3Seg": draw(1 - dropout_rates["seg3_3Seg"][1]),
            "seg4_3Seg": draw(1 - dropout_rates["seg4_3Seg"][1]),
        }
        if two_dof is not None:
            set_to_one = np.logical_and(
                np.logical_and(two_dof[:, None, None], fcs["seg3_3Seg"] == 0.0),
                fcs["seg4_3Seg"] == 0,
            )
            fcs["seg3_3Seg"] = np.where(
                set_to_one, np.ones((bs, 1, 1)), fcs["seg3_3Seg"]
            )
            fcs["seg4_3Seg"] = np.where(
                set_to_one, np.ones((bs, 1, 1)), fcs["seg4_3Seg"]
            )

        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            factor_imu = draw(1 - imu_rate)
            factor_ja = draw(1 - jointaxes_rate)

            if segments in fcs:
                factor_ja = fcs[segments]

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


THREE_SEG_CBS = False
SADDLE_CBS = True


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
    checkpoint: str = None,
    kill_after_hours: float = None,
):
    assert tp is not None

    if tp is not None:
        tp = tp.split(",")

    random.seed(seed)
    np.random.seed(seed)

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_name, config=locals())

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

    # 1SEG exp callbacks
    timings = {
        "S_07": ["slow_fast_mix", "slow_fast_freeze_mix"],
    }

    for anchor_1Seg in ["seg1", "seg5", "seg2", "seg3", "seg4"]:
        sys = ml.convenient.load_1Seg2Seg3Seg4Seg_system(
            anchor_1Seg=anchor_1Seg
        ).change_model_name(suffix=f"_{anchor_1Seg}")
        for exp_id in timings:
            for phase in timings[exp_id]:
                cb = ml.convenient.build_experimental_validation_callback2(
                    rnno_fn,
                    sys,
                    exp_id,
                    phase,
                    rootincl=True,
                    X_transform=natural_units_X_trafo,
                )
                add_callback(cb, sys)

    # 4SEG exp callbacks
    timings = {
        "S_06": ["slow1", "fast"],
        "S_07": ["slow_fast_mix", "slow_fast_freeze_mix"],
    }
    sys = ml.convenient.load_1Seg2Seg3Seg4Seg_system(
        anchor_4Seg="seg5", delete_inner_imus=True
    )
    for exp_id in timings:
        for phase in timings[exp_id]:
            for flex in [False]:
                cb = ml.convenient.build_experimental_validation_callback2(
                    rnno_fn,
                    sys,
                    exp_id,
                    phase,
                    jointaxes=True,
                    rootincl=True,
                    flex=flex,
                    X_transform=natural_units_X_trafo,
                )
                add_callback(cb, sys, twice=True)

    if THREE_SEG_CBS:
        # 3SEG exp callbacks
        sys = ml.convenient.load_1Seg2Seg3Seg4Seg_system(
            anchor_3Seg="seg2", delete_inner_imus=True
        )
        exp_id = "S_07"
        for phase in timings[exp_id]:
            cb = ml.convenient.build_experimental_validation_callback2(
                rnno_fn,
                sys,
                exp_id,
                phase,
                jointaxes=False,
                rootincl=True,
                flex=False,
                X_transform=natural_units_X_trafo,
            )
            add_callback(cb, sys)

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

    def load_sys_flexible(fem, tib, suffix):
        return (
            x_xy.load_example("knee_flexible_imus")
            .change_model_name(suffix="_" + suffix)
            .change_link_name("femur", fem)
            .change_link_name("tibia", tib)
        )

    timings.update(dict(S_16=["gait_slow", "gait_fast"]))
    for exp_id in timings:
        for phase in timings[exp_id]:
            for axis in axes[exp_id]:
                for ja in [True]:
                    sys = load_sys_flexible(*axes[exp_id][axis], suffix=axis)
                    cb = ml.convenient.build_experimental_validation_callback2(
                        rnno_fn,
                        sys,
                        exp_id,
                        phase,
                        jointaxes=ja,
                        flex=True,
                        rootincl=True,
                        X_transform=natural_units_X_trafo,
                    )
                    add_callback(cb, sys)

    del sys

    if SADDLE_CBS:
        callbacks += [
            SaddleCallback(rnno_fn, [0, 0, -1.0], [0, 1.0, 0]),
            SaddleCallback(rnno_fn, [0, 0, 1.0], [0, 0, 0.0]),
            SaddleCallback(rnno_fn, [0, 1, 0.0], [0, 0, 0.0]),
            SaddleCallback(rnno_fn, [0, 0, 0], [0, 1, 0]),
            SaddleCallback(rnno_fn, [0, 0, 0], [0, 0, -1]),
        ]

    # create one large "experimental validation" metric
    for zoom_in in metrices_name:
        print(zoom_in)
    callbacks += [ml.callbacks.AverageMetricesTLCB(metrices_name, "exp_val_mae_deg")]

    sys = ml.convenient.load_1Seg2Seg3Seg4Seg_system(
        "seg2", "seg2", "seg2", "seg5", add_suffix_to_linknames=True
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
        cos_decay_twice=(episodes > 5000),
    )

    ml.train(
        generator,
        episodes,
        network,
        optimizer=optimizer,
        callbacks=callbacks,
        callback_kill_after_seconds=23.5 * 3600
        if kill_after_hours is None
        else kill_after_hours * 3600,
        callback_kill_if_nan=True,
        callback_kill_if_grads_larger=1e32,
        callback_save_params=f"~/params/{ml.unique_id()}.pickle",
        callback_kill_after_episode=kill_ep,
        callback_save_params_track_metrices=[["exp_val_mae_deg"]],
        initial_params=None if warmstart is None else f"~/params/0x{warmstart}.pickle",
        key_network=jax.random.PRNGKey(seed),
        checkpoint=None
        if checkpoint is None
        else f"~/.xxy_checkpoints/0x{checkpoint}.pickle",
    )


if __name__ == "__main__":
    fire.Fire(main)
