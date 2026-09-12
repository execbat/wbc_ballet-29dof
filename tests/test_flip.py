"""CPU semantics and MJCF checks for the flip task."""
import unittest
from types import SimpleNamespace as NS
import torch
import mujoco
import numpy as np

import wbc_ballet.tasks  # noqa: F401  (registers Mjlab-Flip-Flat-Unitree-G1-29DoF)
from wbc_ballet.tasks.flip.mdp import rewards as mdp
from wbc_ballet.tasks.flip.mdp import events as reset
from wbc_ballet.tasks.flip.mdp import terminations as flip_terminations
from mjlab.tasks.registry import load_env_cfg, list_tasks
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

TASK_ID = "Mjlab-Flip-Flat-Unitree-G1-29DoF"


class FlipTests(unittest.TestCase):
    def env(self,n=1):
        command=torch.zeros(n,62)
        scene={'robot':NS(data=NS(root_link_ang_vel_w=torch.zeros(n,3),
            root_link_pos_w=torch.zeros(n,3),
            body_link_quat_w=torch.tensor([1.,0,0,0]).repeat(n,1,1)))}
        class Scene(dict): pass
        scene=Scene(scene);scene.env_origins=torch.zeros(n,3)
        return NS(scene=scene,device='cpu',command_manager=NS(get_command=lambda name: command[:,61:62] if name == 'flip' else command[:,:61])),command

    def test_registration_and_independence(self):
        self.assertIn(TASK_ID,list_tasks())
        c=load_env_cfg(TASK_ID);b=load_env_cfg('Mjlab-Ballet-Flat-Unitree-G1-29DoF')
        self.assertIn('flip',c.observations['actor'].terms)
        self.assertNotIn('flip',b.observations['actor'].terms)
        self.assertIn('commanded_arm_ground_contact',c.rewards)

    def test_flip_orientation_height_and_world_yaw(self):
        env,cmd=self.env(2);cmd[1,61]=1
        env.scene['robot'].data.body_link_quat_w[1,0]=torch.tensor([0.,1.,0.,0.])
        cfg=NS(name='robot',body_ids=[0])
        torch.testing.assert_close(mdp.upright(env,.5,cfg),torch.ones(2))
        env.scene['robot'].data.root_link_pos_w[:,2]=torch.tensor([.77,.58])
        torch.testing.assert_close(mdp.pelvis_height_penalty(env),torch.zeros(2))
        cmd[:,2]=.7;env.scene['robot'].data.root_link_ang_vel_w[:,2]=.7
        torch.testing.assert_close(mdp.track_angular_velocity(env,.5),torch.ones(2))
        cmd[:,2]=-.7
        self.assertTrue((mdp.track_angular_velocity(env,.5)<.01).all())

    def test_contact_truth_table_and_mode_gate(self):
        # all combinations of mode, command movement, masks, and ground contacts
        import itertools
        rows=list(itertools.product((0,1),repeat=6))
        env,cmd=self.env(len(rows));found=[];expected=[]
        for i,(mode,moving,l,r,lc,rc) in enumerate(rows):
            cmd[i,61]=mode;cmd[i,0]=moving;cmd[i,32]=l;cmd[i,33]=r
            found.append([lc,rc])
            val=0
            if not l and not r and not moving:val=2-lc-rc
            elif l and not r:val=lc+(not moving)*(1-rc)
            elif r and not l:val=rc+(not moving)*(1-lc)
            expected.append(val*mode)
        env.scene['hands']=NS(primary_names=['L','R'],data=NS(found=torch.tensor(found)))
        cost=mdp.commanded_contact(env,'hands',('L','R'),NS(joint_ids=[0]),NS(joint_ids=[1]),1)
        torch.testing.assert_close(cost,torch.tensor(expected,dtype=torch.float32))

    def test_support_switch_and_mask(self):
        env,cmd=self.env(3)
        robot=env.scene['robot'];robot.data.heading_w=torch.zeros(3)
        robot.data.site_pos_w=torch.tensor([[[-1.,0,0],[1.,0,0],[-3.,0,0],[3.,0,0]]]).repeat(3,1,1)
        cmd[1:,61]=1;cmd[0,32]=1;cmd[1,34]=1;cmd[2,35]=1
        args=dict(feet_cfg=NS(site_ids=[0,1]),hands_cfg=NS(site_ids=[2,3]),
            left_leg_cfg=NS(joint_ids=[0]),right_leg_cfg=NS(joint_ids=[1]),
            left_arm_cfg=NS(joint_ids=[2]),right_arm_cfg=NS(joint_ids=[3]))
        torch.testing.assert_close(mdp.support_center(env,**args),torch.tensor([[1.,0],[3.,0],[-3.,0]]))

    def test_slip_mode_gate(self):
        env,cmd=self.env(2);cmd[1,61]=1
        env.scene['hands']=NS(primary_names=['L','R'],data=NS(found=torch.ones(2,2)))
        env.scene['robot'].data.site_lin_vel_w=torch.ones(2,2,3)
        value=mdp.slip(env,'hands',('L','R'),NS(name='robot',site_ids=[0,1]),1)
        torch.testing.assert_close(value,torch.tensor([0.,4.]))

    def test_low_pelvis_requires_two_continuous_seconds(self):
        class Scene(dict): pass
        scene=Scene({'robot':NS(data=NS(root_link_pos_w=torch.tensor([[0.,0.,.1]])))})
        scene.env_origins=torch.zeros(1,3)
        env=NS(scene=scene,num_envs=1,device='cpu',step_dt=.5,
               episode_length_buf=torch.tensor([2]))
        for _ in range(3):
            self.assertFalse(flip_terminations.pelvis_height_below_for_duration(env).item())
        self.assertTrue(flip_terminations.pelvis_height_below_for_duration(env).item())
        scene['robot'].data.root_link_pos_w[:,2]=.3
        self.assertFalse(flip_terminations.pelvis_height_below_for_duration(env).item())

    def test_handstand_geometry_and_friction(self):
        cfg=load_env_cfg(TASK_ID)
        entity=Entity(cfg.scene.entities['robot']);m=entity.spec.compile();d=mujoco.MjData(m)
        d.qpos[:7]=[0,0,reset.HANDSTAND_HEIGHT,0,1,0,0]
        for side in ('left','right'):
            for joint,angle in reset.HANDSTAND_LEFT.items():
                if side=='right' and ('roll' in joint or 'yaw' in joint):angle=-angle
                jid=m.joint(f'{side}_{joint}_joint').id
                self.assertTrue(m.jnt_range[jid,0]<angle<m.jnt_range[jid,1])
                d.qpos[m.jnt_qposadr[jid]]=angle
        mujoco.mj_forward(m,d)
        for side in ('left','right'):
            g=m.geom(side+'_hand_collision').id
            # Exact world-Z half-extent of the box support geometry.
            extent=(abs(d.geom_xmat[g].reshape(3,3)[2])*m.geom_size[g]).sum()
            bottom=d.geom_xpos[g,2]-extent
            self.assertTrue(0<=bottom<.02,(side,bottom))
            self.assertEqual(m.geom_condim[g],3)
            self.assertGreater(m.geom_friction[g,0],0)
        # No initial self-penetration in the fitted reference.
        self.assertFalse(any(c.dist < -.001 for c in d.contact))


if __name__=='__main__':unittest.main()
