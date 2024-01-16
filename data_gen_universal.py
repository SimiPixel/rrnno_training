from functools import reduce
import os
from pathlib import Path

import fire
import x_xy
from x_xy.subpkgs import ml


def main(
    configs: list[str],
    size: int,
    non: bool = False,
    prob_rigid: float = 0.5,
    pos_min_max: float = 0.0,
    vault: bool = False,
    all_rigid_or_flex: bool = False,
):
    ENV_VAR = "HPCVAULT" if vault else "WORK"

    root = Path(os.environ.get(ENV_VAR, "")).joinpath("xxy_data")
    root.mkdir(exist_ok=True)
    filepath = root.joinpath(
        f"uni_{size}{reduce(lambda a,b: a+'_'+b, configs, '')}_"
        f"noisy_{int(not non)}_prob_rigid_{str(prob_rigid).replace('.', '')}"
        f"pmm_{str(pos_min_max).replace('.', '')}_allRoF_{int(all_rigid_or_flex)}"
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
                    ml.convenient.load_2Seg3Seg4Seg_system(
                        a2S, a3S, a4S, True, add_suffix_to_linknames=True
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
        ),
        randomize_joint_params=True,
        randomize_motion_artifacts=True,
        randomize_positions=True,
        mode="hdf5",
        hdf5_filepath=filepath,
        sizes=size,
        seed=1,
    )


if __name__ == "__main__":
    fire.Fire(main)
