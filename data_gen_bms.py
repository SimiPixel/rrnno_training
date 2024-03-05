from dataclasses import replace
from functools import reduce
import os
from pathlib import Path

import fire
import jax.numpy as jnp
import x_xy
from x_xy import ml


def finalize_fn_factory(rand_sampling_rates: bool):
    def finalize_fn(key, q, x, sys: x_xy.System):
        X, y = {}, {}

        if rand_sampling_rates:
            X["dt"] = jnp.array([sys.dt], dtype=jnp.float32)

        return X, y

    return finalize_fn


T_global = 60.0


def _duplicate_systems_and_configs(
    sys: list[x_xy.System], configs: list[x_xy.RCMG_Config]
):
    SAMPLING_RATES = [40, 60, 80, 100, 120, 140, 160, 180, 200]
    if not ml.on_cluster():
        SAMPLING_RATES = SAMPLING_RATES

    sys_out, configs_out = [], []
    for _sys in sys:
        for _config in configs:
            for hz in SAMPLING_RATES:
                dt = 1 / hz
                T = (T_global / _sys.dt) * dt

                sys_out.append(_sys.replace(dt=dt))
                configs_out.append(replace(_config, T=T))
    return sys_out, configs_out


def main(
    configs: list[str],
    size: int,
    vault: bool = False,
    flex: bool = False,
    seed: int = 1,
    two_seg: bool = False,
    three_seg: bool = True,
    three_seg_jointaxes: str = None,
    rand_sampling_rates: bool = False,
):
    ENV_VAR = "HPCVAULT" if vault else "WORK"

    pos_min_max: float = 0.05
    add_X_jointaxes = True

    root = Path(os.environ.get(ENV_VAR, "")).joinpath("xxy_data")
    root.mkdir(exist_ok=True)
    filepath = root.joinpath(
        f"uni_{size}{reduce(lambda a,b: a+'_'+b, configs, '')}_seed{seed}_randHz"
        f"_{int(rand_sampling_rates)}_2Seg_{int(two_seg)}_3Seg_{int(three_seg)}"
        f"_flex_{int(flex)}_3SegJAs_{three_seg_jointaxes}"
    )

    configs = [ml.convenient.load_config(name) for name in configs]

    anchors_2Seg = anchors_3Seg = [None]
    replace_joints = False
    if two_seg:
        anchors_2Seg = ["seg2", "seg3"]
    if three_seg:
        anchors_3Seg = ["seg2", "seg4"]
        if three_seg_jointaxes == "yz":
            use_rr_imp = False
            add_X_jointaxes = False
        elif three_seg_jointaxes == "xy":
            use_rr_imp = False
            add_X_jointaxes = False
            replace_joints = True
        else:
            raise Exception(f"Not valid value for jointaxes {three_seg_jointaxes}")

    if not ml.on_cluster():
        anchors_2Seg = anchors_2Seg[:1]
        anchors_3Seg = anchors_3Seg[:1]

    def replace_yz_by_xy(sys: x_xy.System):
        if not replace_joints:
            return sys
        for joint_type, new_joint_type in zip(["ry", "rz"], ["rx", "ry"]):
            name = sys.findall_bodies_with_jointtype(joint_type, names=True)[0]
            sys = sys.change_joint_type(name, new_joint_type)
        return sys

    sys_data = []
    for a2S in anchors_2Seg:
        for a3S in anchors_3Seg:
            sys_data.append(
                replace_yz_by_xy(
                    ml.convenient.load_1Seg2Seg3Seg4Seg_system(
                        None,
                        a2S,
                        a3S,
                        None,
                        use_rr_imp=use_rr_imp,
                        add_suffix_to_linknames=True,
                    )
                )
            )

    zip_sys_config = False
    if rand_sampling_rates:
        sys_data, configs = _duplicate_systems_and_configs(sys_data, configs)
        zip_sys_config = True

    x_xy.build_generator(
        sys_data,
        configs,
        sys_ml=sys_data[0],
        add_X_imus=True,
        add_X_jointaxes=add_X_jointaxes,
        add_X_jointaxes_kwargs=dict(randomly_flip=True),
        add_y_relpose=True,
        add_y_rootincl=True,
        dynamic_simulation=flex,
        imu_motion_artifacts=flex,
        imu_motion_artifacts_kwargs=dict(
            pos_min_max=pos_min_max,
            all_imus_either_rigid_or_flex=True,
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        mode="pickle",
        filepath=filepath,
        sizes=size,
        seed=seed,
        finalize_fn=finalize_fn_factory(rand_sampling_rates),
        jit=ml.on_cluster(),
        zip_sys_config=zip_sys_config,
    )


if __name__ == "__main__":
    fire.Fire(main)
