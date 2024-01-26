from functools import reduce
import os
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import x_xy
from x_xy.subpkgs import ml


def _l_idx_3Seg_last_seg(sys: x_xy.System) -> int:
    lam = sys.link_parents

    root_to_last_map = {"seg2_3Seg": "seg4_3Seg", "seg4_3Seg": "seg2_3Seg"}

    to_root = []
    for options in root_to_last_map:
        if lam[sys.name_to_idx(options)] == -1:
            to_root.append(options)

    assert len(to_root) == 1
    return sys.name_to_idx(root_to_last_map[to_root[0]])


def setup_fn_2DOF_factory(prob: float | None):
    if prob is None:
        return None

    def setup_fn_2DOF(key, sys: x_xy.System):
        flip = jax.random.bernoulli(key, p=prob)

        def collapse(sys: x_xy.System):
            # print("collapse")
            pos = sys.links.transform1.pos
            new_pos = pos.at[_l_idx_3Seg_last_seg(sys)].set(jnp.zeros((3,)))
            return sys.replace(
                links=sys.links.replace(
                    transform1=sys.links.transform1.replace(pos=new_pos)
                )
            )

        return jax.lax.cond(flip, collapse, lambda sys: sys, sys)

    return setup_fn_2DOF


def finalize_fn_factory(prob: float | None):
    def finalize_fn(key, q, x, sys):
        X, y = {}, {}
        pos_i = sys.links.transform1.pos[_l_idx_3Seg_last_seg(sys)]
        X["2DOF"] = jnp.allclose(pos_i, jnp.zeros((3,)))
        return X, y

    return None if prob is None else finalize_fn


def main(
    configs: list[str],
    size: int,
    non: bool = False,
    prob_rigid: float = 0.5,
    pos_min_max: float = 0.0,
    vault: bool = False,
    all_rigid_or_flex: bool = False,
    prob_2DOF: float = None,
    seed: int = 1,
):
    ENV_VAR = "HPCVAULT" if vault else "WORK"

    root = Path(os.environ.get(ENV_VAR, "")).joinpath("xxy_data")
    root.mkdir(exist_ok=True)
    filepath = root.joinpath(
        f"uni_{size}{reduce(lambda a,b: a+'_'+b, configs, '')}_"
        f"noisy_{int(not non)}_prob_rigid_{str(prob_rigid).replace('.', '')}"
        f"pmm_{str(pos_min_max).replace('.', '')}_allRoF_{int(all_rigid_or_flex)}"
        f"_2DOF_{str(prob_2DOF).replace('.', '')}_seed_{seed}"
    )

    configs = [ml.convenient.load_config(name) for name in configs]

    anchors_2Seg = anchors_3Seg = anchors_4Seg = [None]
    anchors_2Seg = ["seg2", "seg3"]
    anchors_3Seg = ["seg2", "seg4"]
    anchors_4Seg = ["seg5", "seg2", "seg3", "seg4"]

    if not ml.on_cluster():
        anchors_2Seg = anchors_2Seg[:1]
        anchors_3Seg = anchors_3Seg[:1]
        anchors_4Seg = anchors_4Seg[:1]

    sys_data = []
    for a2S in anchors_2Seg:
        for a3S in anchors_3Seg:
            for a4S in anchors_4Seg:
                sys_data.append(
                    ml.convenient.load_1Seg2Seg3Seg4Seg_system(
                        "seg2", a2S, a3S, a4S, True, add_suffix_to_linknames=True
                    )
                )

    x_xy.build_generator(
        sys_data,
        configs,
        sys_ml=sys_data[0],
        add_X_imus=True,
        add_X_imus_kwargs=dict(noisy=not non),
        add_X_jointaxes=True,
        add_y_relpose=True,
        add_y_rootincl=True,
        dynamic_simulation=True,
        imu_motion_artifacts=True,
        imu_motion_artifacts_kwargs=dict(
            prob_rigid=prob_rigid,
            pos_min_max=pos_min_max,
            all_imus_either_rigid_or_flex=all_rigid_or_flex,
            imus_surely_rigid=["imu2_1Seg"],
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        mode="hdf5",
        hdf5_filepath=filepath,
        sizes=size,
        seed=seed,
        setup_fn=setup_fn_2DOF_factory(prob_2DOF),
        finalize_fn=finalize_fn_factory(prob_2DOF),
        jit=ml.on_cluster(),
    )


if __name__ == "__main__":
    fire.Fire(main)
