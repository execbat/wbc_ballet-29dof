import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import socket
import unittest
import numpy as np
import torch

from wbc_ballet.tasks.flip.mdp import commands
from wbc_ballet.teleop import flip_protocol as protocol
from wbc_ballet.tasks.flip.mdp import rewards as mdp
from mjlab.tasks.registry import load_env_cfg


class FlipCommandTests(unittest.TestCase):
    def test_independent_timer_and_random_binary_values(self):
        env=NS(num_envs=256,device='cpu',_flip_mode=torch.zeros(256))
        term=commands.FlipCommandCfg(probability=1.,resampling_time_range=(12.,12.)).build(env)
        term.reset(torch.arange(256))
        self.assertTrue((term.command==0).all())
        term.compute(11.)
        self.assertTrue((term.command==0).all())
        term.compute(1.)
        self.assertTrue((term.command==1).all())
        term.cfg.probability=0.
        term.compute(12.)
        self.assertTrue((term.command==0).all())
        term.cfg.probability=.5
        torch.manual_seed(99)
        term.compute(12.)
        self.assertTrue(((term.command==0)|(term.command==1)).all())
        self.assertTrue(0<term.command.sum()<256)
        # Partial resets do not change another environment's timer/command.
        before=term.command[1:].clone();time=term.time_left[1:].clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.command[1:],before)
        torch.testing.assert_close(term.time_left[1:],time)

    def test_config_has_separate_terms_and_strong_penalty(self):
        cfg=load_env_cfg('Mjlab-Flip-Flat-Unitree-G1-29DoF')
        self.assertEqual(list(cfg.commands),['flip','ballet'])
        lo,hi=cfg.commands['flip'].resampling_time_range
        self.assertEqual((lo,hi),(9.,11.))
        self.assertEqual(cfg.commands['flip'].probability,.5)
        self.assertEqual(cfg.events['reset_robot'].params['flip_probability'],.5)
        self.assertTrue(cfg.events['reset_robot'].params['randomize_pose'])
        self.assertIsNone(cfg.curriculum.get('flip_spawn_probability'))
        self.assertEqual(cfg.rewards['forbidden_support'].weight,-20.)
        self.assertEqual(cfg.rewards['head_ground_contact'].weight,-20.)
        self.assertEqual(cfg.events['hand_friction'].params['ranges'],(.3,1.2))
        self.assertEqual(cfg.events['foot_friction'].params['ranges'],(.3,1.2))
        self.assertEqual(list(cfg.actions),['joint_pos'])
        self.assertEqual(cfg.terminations['fell_over'].params['duration_s'],2.)
        play=load_env_cfg('Mjlab-Flip-Flat-Unitree-G1-29DoF',play=True)
        self.assertIsInstance(play.commands['flip'],commands.UdpFlipCommandCfg)
        self.assertIsInstance(play.commands['ballet'],commands.UdpFlipBalletCommandCfg)
        self.assertIsNone(play.events['reset_robot'].params['flip_probability'])
        self.assertTrue(play.events['reset_robot'].params['randomize_pose'])

    def test_wrong_support_immediate_both_directions(self):
        mode=torch.zeros(3,1)
        feet=NS(primary_names=['right_ankle_roll_link','left_ankle_roll_link'],
                data=NS(found=torch.tensor([[1,1],[0,0],[1,0]])))
        hands=NS(primary_names=['right_wrist_yaw_link','left_wrist_yaw_link'],
                 data=NS(found=torch.tensor([[0,0],[1,1],[0,1]])))
        env=NS(scene={'feet_ground_contact':feet,'hands_ground_contact':hands},
               command_manager=NS(get_command=lambda _:mode))
        torch.testing.assert_close(mdp.forbidden_support(env),torch.tensor([0.,2.,1.]))
        mode[:]=1
        torch.testing.assert_close(mdp.forbidden_support(env),torch.tensor([2.,0.,1.]))
        mode[:]=0
        torch.testing.assert_close(mdp.forbidden_support(env),torch.tensor([0.,2.,1.]))

    def test_udp_packet_coherent_across_two_terms(self):
        env=NS(num_envs=2,device='cpu')
        flip=commands.UdpFlipCommandCfg(port=0).build(env)
        ballet=commands.UdpFlipBalletCommandCfg(port=0).build(env)
        self.assertIs(flip.receiver,ballet.receiver)
        tx=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            addr=flip.receiver.socket.getsockname()
            for value in (0,1,0):
                vec=np.zeros(62,dtype='<f4');vec[0]=.35;vec[29]=1;vec[58:61]=[.1,-.2,.7];vec[61]=value
                tx.sendto(vec.tobytes(),addr)
                flip.compute(.02);ballet.compute(.02)
                self.assertTrue((flip.command==value).all())
                torch.testing.assert_close(ballet.command[0,:3],torch.tensor([.1,-.2,.7]))
                self.assertAlmostEqual(ballet.command[0,3].item(),.35,places=6)
                self.assertEqual(ballet.command.shape,(2,61))
            for bad in (np.zeros(61,dtype='<f4'),np.full(62,np.nan,dtype='<f4')):
                tx.sendto(bad.tobytes(),addr)
                flip.compute(.02);ballet.compute(.02)
                self.assertTrue((flip.command==0).all())
            vec[61]=.5
            with self.assertRaises(ValueError):protocol.FlipPacket.from_bytes(vec.tobytes())
        finally:
            tx.close();flip.receiver.close()

    def test_gamepad_live_and_pattern_flip(self):
        # Find the v2 gamepad in the delivered repo layout, without launching Tk.
        here=Path(__file__).resolve()
        path=next((parent/'gamepad/game_emulator_run_v2.py' for parent in here.parents
                   if (parent/'gamepad/game_emulator_run_v2.py').exists()),None)
        self.assertIsNotNone(path)
        spec=importlib.util.spec_from_file_location('gamepad_v2_test',path)
        gui=importlib.util.module_from_spec(spec);spec.loader.exec_module(gui)
        def var(value):return NS(get=lambda:value)
        app=NS(slider_vars=[var(.25)]*29,switch_vars=[var(1)]*29,
               speed_x_var=var(.1),speed_y_var=var(.2),angle_z_var=var(.3),
               pattern_targets=np.full((1,29),.25,dtype=np.float32),pattern_masks=np.ones((1,29),dtype=np.float32))
        try:
            for value in (0,1,0):
                app.flip_var=var(value)
                for vec in (gui.AxisControlApp._collect_live_vector(app),
                            gui.AxisControlApp._collect_pattern_vector(app,0,sync_ui=False)):
                    self.assertEqual(vec.shape,(62,))
                    packet=protocol.FlipPacket.from_bytes(vec.tobytes())
                    self.assertEqual(packet.flip,value)
                    np.testing.assert_allclose(packet.velocity,[.1,.2,.3],rtol=1e-6)
        finally:gui.sock.close()


if __name__=='__main__':unittest.main()
