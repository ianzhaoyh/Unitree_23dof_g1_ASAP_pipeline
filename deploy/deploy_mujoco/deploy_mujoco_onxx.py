import time
import mujoco.viewer
import mujoco
import numpy as np
import onnxruntime
import yaml
import argparse
import math

LEGGED_GYM_ROOT_DIR = "/home/mycode/ASAP"

class ObsBuilder:
    def __init__(self, config):
        # 历史观测配置
        self.dof_pos_scale = config['dof_pos_scale']
        self.dof_vel_scale = config['dof_vel_scale']
        self.ang_vel_scale = config['ang_vel_scale']
        self.default_angles = config['default_angles']
        self.num_actions = config['num_actions']  #21
        self.num_obs = config['num_obs']   #70
        self.history_length = config['history_length']  # 280/70=4 steps
        self.history_actor = np.zeros((self.history_length, self.num_obs), dtype=np.float32)
        self.history_step = 0
        

        self.simulation_dt = config['simulation_dt']
        # Controller update frequency (meets the requirement of simulation_dt * controll_decimation=0.02; 50Hz)
        self.control_decimation = config['control_decimation']
        # self.ref_period = self.simulation_dt * self.control_decimation
        self.phase = 0


        
    def _update_history(self, new_obs):
        """更新历史观测缓冲区（环形缓冲区）"""
        idx = self.history_step % self.history_length
        self.history_actor[idx] = new_obs
        self.history_step += 1
        
    def get_actor_obs(self, d, last_action):
        """构建350维actor观测向量"""
        # 基础观测
        base_ang_vel = d.qvel[3:6] * self.ang_vel_scale  # [3]
        projected_gravity = get_gravity_orientation(d.qpos[3:7])  # [3]
        
        # 关节相关
        dof_pos = (d.qpos[7: self.num_actions +7] - self.default_angles) * self.dof_pos_scale  # [21]
        dof_vel = d.qvel[6: self.num_actions +6] * self.dof_vel_scale  # [21]
        
        # 相位生成（基于仿真时间）
        # phase = (d.time % self.ref_period) / self.ref_period  # [1]
        self.phase += 0.00325
        
        # 当前观测片段（用于历史记录）
        current_obs_segment = np.concatenate((
            last_action,
            base_ang_vel,
            
            dof_pos,
            dof_vel,
            
            projected_gravity,
            [self.phase]
         ), axis=-1, dtype=np.float32)
        
        # 更新历史（前9步历史+当前）
        if self.history_step >= self.history_length:
            self._update_history(current_obs_segment)
            valid_history = self.history_actor
        else:
            # 初始阶段填充历史
            valid_history = np.vstack([
                self.history_actor[:self.history_step],
                np.tile(current_obs_segment, 
                       (self.history_length - self.history_step, 1))
            ])
        history_obs_buf = np.concatenate((action_buf, ang_vel_buf, dof_pos_buf, dof_vel_buf, proj_g_buf, ref_motion_phase_buf), axis=-1, dtype=np.float32)
        
        obs_buf = np.concatenate((action, base_ang_vel, dof_pos, dof_vel, history_obs_buf, projected_gravity, np.array([ref_motion_phase])), axis=-1, dtype=np.float32)
                    
        
        return np.concatenate((
            last_action,                # 21
            base_ang_vel,               # 3
            
            dof_pos,                    # 21
            dof_vel,                    # 21
            valid_history.flatten(),     # 280
            projected_gravity,          # 3
            [self.phase]                    # 1
            
        ),axis=-1,  dtype=np.float32)            # Total 350

def get_gravity_orientation(quaternion):
    # 保持原实现不变
    qw, qx, qy, qz = quaternion
    return np.array([
        2*(-qz*qx + qw*qy),
        -2*(qz*qy + qw*qx),
        1 - 2*(qw**2 + qz**2)
    ])

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        


    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    obs_builder = ObsBuilder(config)
    
    # 加载ONNX模型
    ort_session = onnxruntime.InferenceSession(
        config['policy_path'],
        providers=['CUDAExecutionProvider']
    )
    input_name = ort_session.get_inputs()[0].name

    # 主循环


    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        last_action = np.zeros(num_actions , dtype=np.float32)
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:num_actions+7], kps, np.zeros(num_actions), d.qvel[6:num_actions+6], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # 构建观测
                actor_obs = obs_builder.get_actor_obs(d, last_action)
                


                action = np.squeeze(ort_session.run(None, {input_name: actor_obs[None, :]})[0])
                # action = ort_session.run(None, {input_name: actor_obs[None, :]})[0][0]
                last_action = action.copy()
                # action = np.clip(action, -100, 100)
                # print(action)
                # transform action to target_dof_pos
                target_dof_pos = action * 0.25 + default_angles
                
                # target_dof_pos = np.clip(target_dof_pos, - dof_pos_limit, dof_pos_limit)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


