"""Real play env + localhost UDP check for the flip task, default zero and
live flip in both directions. Manual smoke test -- not collected by pytest;
run directly:

    uv run python scripts/smoke_flip_udp.py
"""
import socket
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import wbc_ballet.tasks  # noqa: F401  (registers Mjlab-Flip-Flat-Unitree-G1-29DoF)

cfg=load_env_cfg('Mjlab-Flip-Flat-Unitree-G1-29DoF',play=True)
# An OS-selected port keeps this test separate from a running gamepad.
for term in cfg.commands.values():term.port=0
env=ManagerBasedRlEnv(cfg,device='cpu')
tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
try:
    obs,_=env.reset(seed=42)
    assert env.command_manager.get_command('flip').item()==0
    addr=env._flip_udp_receiver.socket.getsockname()
    for value in (1,0):
        vec=np.zeros(62,dtype='<f4');vec[61]=value;vec[60]=.4
        tx.sendto(vec.tobytes(),addr)
        before=env.scene['robot'].data.root_link_pos_w.clone()
        env.command_manager.compute(dt=100.)
        assert env.command_manager.get_command('flip').item()==value
        assert abs(env.command_manager.get_command('ballet')[0,2].item()-.4)<1e-6
        torch.testing.assert_close(before,env.scene['robot'].data.root_link_pos_w)
        obs,reward,*_=env.step(torch.zeros(1,29))
        assert obs['actor'][0,-1].item()==value
        assert torch.isfinite(reward).all()
        print('PASS UDP play flip',value,'without teleport or random override')
finally:
    tx.close();env._flip_udp_receiver.close();env.close()
