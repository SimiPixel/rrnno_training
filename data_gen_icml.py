from functools import reduce
import os
from pathlib import Path

from configs import load_config
import fire
import x_xy

prob_rigid: float = 0.25
pos_min_max: float = 0.05
all_rigid_or_flex: bool = True
rand_sampling_rates: bool = True


def main(
    configs: list[str],
    size: int,
    vault: bool = False,
    seed: int = 1,
    dry_run: bool = False,
):
    ENV_VAR = "HPCVAULT" if vault else "WORK"

    root = Path(os.environ.get(ENV_VAR, "")).joinpath("xxy_data")
    root.mkdir(exist_ok=True)
    filepath = root.joinpath(
        f"icml_{size}{reduce(lambda a,b: a+'_'+b, configs, '')}_seed_{seed}"
    )

    configs = [load_config(name) for name in configs]
    anchors = [
        "seg2_2Seg",
        "seg3_2Seg",
        "seg2_3Seg",
        "seg4_3Seg",
        "seg5_4Seg",
        "seg2_4Seg",
        "seg3_4Seg",
        "seg4_4Seg",
    ]
    sampling_rates = [40, 60, 80, 100, 120, 140, 160, 180, 200]

    if not x_xy.ml.on_cluster() or dry_run:
        anchors = anchors[:1]
        sampling_rates = [80, 100]

    x_xy.RCMG(
        x_xy.io.load_example("exclude/standard_sys_rr_imp"),
        configs,
        add_X_imus=True,
        add_X_jointaxes=True,
        add_X_jointaxes_kwargs=dict(randomly_flip=True),
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
        randomize_anchors=True,
        randomize_anchors_kwargs=dict(anchors=anchors),
        randomize_hz=rand_sampling_rates,
        randomize_hz_kwargs=dict(sampling_rates=sampling_rates),
    ).to_pickle(filepath, size, seed)


if __name__ == "__main__":
    fire.Fire(main)
