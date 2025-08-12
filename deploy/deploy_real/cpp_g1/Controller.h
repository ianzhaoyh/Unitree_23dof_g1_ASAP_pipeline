#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <onnxruntime_cxx_api.h>
#include "torch/script.h"

#include <eigen3/Eigen/Eigen>

#include "joystick.h"
#include "DataBuffer.h"
#include <string>

class Controller
{
	public:
		Controller(const std::string &net_interface);
		void low_state_message_handler(const void *message);
		void move_to_default_pos();
		void run();
		void damp();
		void zero_torque_state();

	private:




		void low_cmd_write_handler();

		Ort::Env env;
		Ort::SessionOptions session_options;
		Ort::AllocatorWithDefaultOptions allocator;
		// Ort::MemoryInfo memory_info;
		std::unique_ptr<Ort::Session> session;
		
		std::vector<const char*> input_names = {"actor_obs"};
		std::vector<const char*> output_names = {"action"};

		unitree::common::ThreadPtr low_cmd_write_thread_ptr;

		DataBuffer<unitree_hg::msg::dds_::LowCmd_> mLowCmdBuf;
		DataBuffer<unitree_hg::msg::dds_::LowState_> mLowStateBuf;

		unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher;
		unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber;

		// joystick
		xRockerBtnDataStruct joy;

		// yaml config
		std::vector<float> leg_joint2motor_idx;
		std::vector<float> kps;
		std::vector<float> kds;
		std::vector<float> fixed_kps;
		std::vector<float> fixed_kds;

		std::vector<float> default_angles;
		std::vector<float> fixed_target_angles;
		std::vector<float> arm_waist_joint2motor_idx;

		std::vector<float> fixed_joint2motor_idx;

		std::vector<float> arm_waist_kps;
		std::vector<float> arm_waist_kds;
		std::vector<float> arm_waist_target;
		float ang_vel_scale;
		float dof_pos_scale;
		float dof_vel_scale;
		float action_scale;
		std::vector<float> cmd_scale;
		// float num_actions;
		// float num_obs;
		int num_actions;
		int num_obs;
		std::vector<float> max_cmd;

		Eigen::VectorXf obs;
		Eigen::VectorXf act;

		//增加数据限制
		// 例如：关节角度限制
		std::vector<float> dof_lower_limits;
        std::vector<float> dof_upper_limits;

		//当前观测
 		std::vector<float> action;  
 		std::vector<float> ang_vel;  
 		std::vector<float> dof_pos;  
		std::vector<float> dof_vel;  
 		std::vector<float> gravity_orientation;  
 		// std::vector<float> ref_motion_phase;  	

		//历史观测
 		std::vector<float> action_buf;  
 		std::vector<float> ang_vel_buf;  
 		std::vector<float> dof_pos_buf;  
		std::vector<float> dof_vel_buf;  
 		std::vector<float> proj_g_buf;  
 		std::vector<float> ref_motion_phase_buf;  


		// torch::jit::script::Module module;
};

#endif
