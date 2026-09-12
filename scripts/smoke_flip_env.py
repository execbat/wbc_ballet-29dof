"""Real MJLab reset and stepping check for the flip task, without PPO or a
trained policy. Manual smoke test -- not collected by pytest; run directly:

    uv run python scripts/smoke_flip_env.py
"""
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import wbc_ballet.tasks  # noqa: F401  (registers Mjlab-Flip-Flat-Unitree-G1-29DoF)

cfg=load_env_cfg('Mjlab-Flip-Flat-Unitree-G1-29DoF')
cfg.scene.num_envs=4
cfg.events['reset_robot'].params['flip_probability']=.5
env=ManagerBasedRlEnv(cfg,device='cpu')
try:
    obs,_=env.reset(seed=42)
    assert obs['actor'].shape==(4,187)
    assert obs['critic'].shape==(4,216)
    # Check actual per-world model friction after startup randomization.
    robot=env.scene['robot']
    local_ids,_=robot.find_geoms(('left_hand_collision','right_hand_collision'))
    global_ids=robot.indexing.geom_ids[local_ids].long()
    mu=env.sim.model.geom_friction[:,global_ids,0]
    assert ((mu>=.3)&(mu<=1.2)).all(),mu
    assert torch.unique(mu[:,0]).numel()>1,mu
    torch.testing.assert_close(mu[:,0],mu[:,1])
    print('PASS randomized hand friction',mu.tolist())
    for mode in (0.,1.):
        # Reset event settings, not a teleport inside the command/reward.
        env.event_manager.get_term_cfg('reset_robot').params['flip_probability']=mode
        obs,_=env.reset()
        cmd=env.command_manager.get_command('flip')
        assert torch.all(cmd[:,0]==mode),cmd[:,0]
        for _ in range(3):
            obs,reward,terminated,truncated,extras=env.step(torch.zeros(4,29))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(x).all() for x in obs.values())
        # Exercise a live flip resample and mask repair without reset/teleport.
        flip=env.command_manager.get_term('flip')
        ballet=env.command_manager.get_term('ballet')
        # Both pairs fully masked before the change: the new support pair must be repaired.
        ballet.command[:,32:61]=1.
        flip.cfg.probability=1.-mode
        flip.time_left[:]=0.
        ballet.time_left[:]=5.
        qpos=robot.data.joint_pos.clone()
        pose=robot.data.root_link_pos_w.clone()
        env.command_manager.compute(dt=env.step_dt)
        assert torch.all(flip.command==1.-mode)
        torch.testing.assert_close(robot.data.joint_pos,qpos)
        torch.testing.assert_close(robot.data.root_link_pos_w,pose)
        selected=ballet._limb_groups[int(1.-mode)][1]
        mask=ballet.command[:,32:61]
        assert not ((mask[:,selected[0]]>=.5).any(1)&(mask[:,selected[1]]>=.5).any(1)).any()
        obs,reward,*_=env.step(torch.zeros(4,29))
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(x).all() for x in obs.values())
        print('PASS live flip change',mode,'->',1.-mode,'without pose writes')
        print('PASS reset/step mode',mode,'actor',tuple(obs['actor'].shape),'critic',tuple(obs['critic'].shape))
finally:
    env.close()
