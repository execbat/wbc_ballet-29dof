"""Command manager terms for random training and UDP play."""

from wbc_ballet.utils.configclass import configclass

from . import mdp as flip_mdp


@configclass
class FlipCommandsCfg:
    flip: flip_mdp.FlipCommandCfg | None = flip_mdp.FlipCommandCfg(
        probability=0.5,
        resampling_time_range=(9.0, 11.0),
    )
    ballet: flip_mdp.FlipBalletCommandCfg | None = flip_mdp.FlipBalletCommandCfg(
        resampling_time_range=(6.0, 10.0),
        # Pose/mask curricula remain symmetric across the two flip modes.
        target_scale=0.0,
        target_limit=0.8,
        mask_probability=0.0,
        velocity_ranges=((-0.4, -0.3, -1.0), (0.4, 0.3, 1.0)),
    )


@configclass
class FlipUdpCommandsCfg(FlipCommandsCfg):
    flip: flip_mdp.FlipCommandCfg | None = flip_mdp.UdpFlipCommandCfg(
        host="127.0.0.1",
        port=55002,
    )
    ballet: flip_mdp.FlipBalletCommandCfg | None = flip_mdp.UdpFlipBalletCommandCfg(
        host="127.0.0.1",
        port=55002,
    )
