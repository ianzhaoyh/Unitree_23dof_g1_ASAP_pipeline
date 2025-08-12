#include "Controller.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <thread>
#include "utilities.h"
#include <fstream> 
#include <vector>
#include <onnxruntime_cxx_api.h> 
#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"


void update_buffer(std::vector<float>& buffer, const std::vector<float>& new_data, int single_len) 
{
    int total_len = buffer.size();
    int new_len = new_data.size();

    // 向后移动已有数据
    for (int i = total_len - 1; i >= new_len; --i) {
        buffer[i] = buffer[i - new_len];
    }

    // 复制新的数据到前面
    for (int i = 0; i < new_len; ++i) {
        buffer[i] = new_data[i];
    }
}

Controller::Controller(const std::string &net_interface)
    : action(21, 0.0f),
      ang_vel(3, 0.0f),
      dof_pos(21, 0.0f),
      dof_vel(21, 0.0f),
      gravity_orientation(3, 0.0f),
      action_buf(21*4, 0.0f),
      ang_vel_buf(3*4, 0.0f),
      dof_pos_buf(21*4, 0.0f),
      dof_vel_buf(21*4, 0.0f),
      proj_g_buf(3*4, 0.0f),
      ref_motion_phase_buf(1*4, 0.0f)
{

	
        YAML::Node yaml_node = YAML::LoadFile("../../configs/g1.yaml");
    //变量要在.文件声明，任何包含这个头文件的 .cpp 文件，都知道这个类有这个成员。
	leg_joint2motor_idx = yaml_node["leg_joint2motor_idx"].as<std::vector<float>>();
        kps = yaml_node["kps"].as<std::vector<float>>();
        kds = yaml_node["kds"].as<std::vector<float>>();
		fixed_kps = yaml_node["fixed_kps"].as<std::vector<float>>();
		fixed_kds = yaml_node["fixed_kds"].as<std::vector<float>>();

	default_angles = yaml_node["default_angles"].as<std::vector<float>>();
	fixed_target_angles = yaml_node["fixed_target_angles"].as<std::vector<float>>();	
	arm_waist_joint2motor_idx = yaml_node["arm_waist_joint2motor_idx"].as<std::vector<float>>();



	fixed_joint2motor_idx= yaml_node["fixed_joint2motor_idx"].as<std::vector<float>>();


	arm_waist_kps = yaml_node["arm_waist_kps"].as<std::vector<float>>();
	arm_waist_kds = yaml_node["arm_waist_kds"].as<std::vector<float>>();
	arm_waist_target = yaml_node["arm_waist_target"].as<std::vector<float>>();
	ang_vel_scale = yaml_node["ang_vel_scale"].as<float>();
	dof_pos_scale = yaml_node["dof_pos_scale"].as<float>();
	dof_vel_scale = yaml_node["dof_vel_scale"].as<float>();
	action_scale = yaml_node["action_scale"].as<float>();
	cmd_scale = yaml_node["cmd_scale"].as<std::vector<float>>();
	num_actions = yaml_node["num_actions"].as<float>();
	num_obs = yaml_node["num_obs"].as<float>();
	max_cmd = yaml_node["max_cmd"].as<std::vector<float>>();

    dof_lower_limits = yaml_node["dof_lower_limits"].as<std::vector<float>>();
    dof_upper_limits = yaml_node["dof_upper_limits"].as<std::vector<float>>();

    if (dof_lower_limits.size() != dof_upper_limits.size()) {
        std::cerr << "[ERR] dof limit size mismatch "
                  << dof_lower_limits.size() << " vs "
                  << dof_upper_limits.size() << std::endl;
    }
	
	// obs.setZero(num_obs);
	// act.setZero(num_actions);
	obs = Eigen::VectorXf::Zero(num_obs);
	act = Eigen::VectorXf::Zero(num_actions);

	// //当前观测
	// std::vector<float> action(21 , 0.0f);
	// std::vector<float> ang_vel(3 , 0.0f);
	// std::vector<float> dof_pos(21 , 0.0f);
	// std::vector<float> dof_vel(21 , 0.0f);
	// std::vector<float> gravity_orientation(3 , 0.0f);
	// // std::vector<float> ref_motion_phase(1 * 4, 0.0f);
	// //历史观测

	// std::vector<float> action_buf(21 * 4, 0.0f);
	// std::vector<float> ang_vel_buf(3 * 4, 0.0f);
	// std::vector<float> dof_pos_buf(21 * 4, 0.0f);
	// std::vector<float> dof_vel_buf(21 * 4, 0.0f);
	// std::vector<float> proj_g_buf(3 * 4, 0.0f);
	// std::vector<float> ref_motion_phase_buf(1 * 4, 0.0f);


	// module = torch::jit::load("../../../pre_train/g1/motion.pt");

	// ONNX Runtime session options
	env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "inference");
	session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency()); 
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	std::string model_path = "../../../pre_train/g1/model_2000.onnx";
	session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
	// memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	// session = Ort::Session(env, model_path.c_str(), session_options);


	unitree::robot::ChannelFactory::Instance()->Init(0, net_interface);

	lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
	lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(TOPIC_LOWSTATE));

	lowcmd_publisher->InitChannel();
	lowstate_subscriber->InitChannel(std::bind(&Controller::low_state_message_handler, this, std::placeholders::_1));

	while (!mLowStateBuf.GetDataPtr())
	{
		usleep(100000);
	}
	
	low_cmd_write_thread_ptr = unitree::common::CreateRecurrentThreadEx("low_cmd_write", UT_CPU_ID_NONE, 2000, &Controller::low_cmd_write_handler, this);
	std::cout << "Controller init done!\n";
}

void Controller::zero_torque_state()
{
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	std::cout << "zero_torque_state, press start\n";
	while (!joy.btn.components.start)
	{
		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();

		for (auto &cmd : low_cmd->motor_cmd())
		{
			cmd.q() = 0;
			cmd.dq() = 0;
			cmd.kp() = 0;
			cmd.kd() = 0;
			cmd.tau() = 0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::move_to_default_pos()
{
	std::cout << "move_to_default_pos, press A\n";
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	auto low_state = mLowStateBuf.GetDataPtr();	
	std::array<float, 35> jpos;
	for (int i = 0; i < 35; i++)
	{
		jpos[i] = low_state->motor_state()[i].q();
	}

	int num_steps = 100;
	int count = 0;

	while (count <= num_steps || !joy.btn.components.A) 
	{
		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();
		// unitree_hg::msg::dds_::LowCmd_ low_cmd; 
		// size_t motor_count = low_cmd.motor_cmd().size(); 
		// std::cout << "Motor count: " << motor_count << std::endl;

		float phase = std::clamp<float>(float(count++) / num_steps, 0, 1);
		
		// leg
		for (int i = 0; i < 12; i++)
		{
			low_cmd->motor_cmd()[i].q() = (1 - phase) * jpos[i] + phase * default_angles[i];
			low_cmd->motor_cmd()[i].kp() = kps[i];
			low_cmd->motor_cmd()[i].kd() = kds[i];
			low_cmd->motor_cmd()[i].tau() = 0.0;
			low_cmd->motor_cmd()[i].dq() = 0.0;
		}

		// waist arm
		for (int i = 12; i < 29; i++)
		{
			low_cmd->motor_cmd()[i].q() = (1 - phase) * jpos[i] + phase * arm_waist_target[i - 12];
			low_cmd->motor_cmd()[i].kp() = arm_waist_kps[i - 12];
			low_cmd->motor_cmd()[i].kd() = arm_waist_kds[i - 12];
			low_cmd->motor_cmd()[i].tau() = 0.0;
			low_cmd->motor_cmd()[i].dq() = 0.0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::run()
{

	std::cout << "obs.size() = " << obs.size() << std::endl;
	std::cout << "act.size() = " << act.size() << std::endl;

    std::cout << "Pre-warming the model..." << std::endl;
    {
        Eigen::VectorXf dummy_obs = Eigen::VectorXf::Zero(obs.size());
        std::vector<int64_t> input_shape = {1, dummy_obs.size()};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, dummy_obs.data(), dummy_obs.size(), input_shape.data(), input_shape.size());
        for (int i = 0; i < 50; ++i) {
            session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        }
    }
    std::cout << "Pre-warming complete." << std::endl;

	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();
std::cout << "[DEBUG] obs.size()=" << obs.size()
          << ", act.size()=" << act.size() << std::endl;
	float period = 17.433;  //robominic
	float step_count = 0.0f;
	int flag = 0;
	while (!joy.btn.components.select)
	{
		auto low_state = mLowStateBuf.GetDataPtr();
		std::cout << 1 << std::endl;
		// obs
		Eigen::Matrix3f R = Eigen::Quaternionf(low_state->imu_state().quaternion()[0], low_state->imu_state().quaternion()[1], low_state->imu_state().quaternion()[2], low_state->imu_state().quaternion()[3]).toRotationMatrix();
		std::cout << 2 << std::endl;



////////////////

	// action #  21
    // base_ang_vel # 3
    // dof_pos # 21
    // dof_vel # 21
    // history_actor # 4 * (21+3+21+21+3+1)=4*70=280
    // projected_gravity # 3
    // ref_motion_phase # 1 
    // 280+70=350

///////////////		
		size_t motor_count = low_state->motor_state().size();
		std::cout << "motor_count = " << motor_count << std::endl;
        // 打印所有电机的状态数据
        for (size_t i = 0; i < motor_count; ++i) {
            const auto& motor = low_state->motor_state()[i];
            std::cout << "  Motor[" << i << "]: "
                      << "q=" << motor.q() 
                      << ", dq=" << motor.dq() 

                      << std::endl;
        }
		// for (int i = 0; i < motor_count; i++) 
		// {
		// 	dof_pos[i] = low_state->motor_state()[i].q() 
		// }


		obs.segment(0, 21) = act;
		std::cout << 13<< std::endl;
		//20
		for (int i = 0; i < 3; i++)
		{
			ang_vel[i] = ang_vel_scale * low_state->imu_state().gyroscope()[i];
			obs(21+i) = ang_vel_scale * low_state->imu_state().gyroscope()[i];
		}
		//23
		for (int i = 0; i < 13; i++) //12个腿部关节+1个腰部关节
		{
			dof_pos[i] = (low_state->motor_state()[i].q() - default_angles[i]) * dof_pos_scale;
			dof_vel[i] = low_state->motor_state()[i].dq() * dof_vel_scale;

			obs(24 + i) = (low_state->motor_state()[i].q() - default_angles[i]) * dof_pos_scale;//36
			obs(45 + i) = low_state->motor_state()[i].dq() * dof_vel_scale;  //57
		}
		//36

		for (int i = 0; i < 4; i++) //4个左手部关节
		{
			dof_pos[13+i] = (low_state->motor_state()[15+i].q() - default_angles[15+i]) * dof_pos_scale;
			dof_vel[13+i] = low_state->motor_state()[15+i].dq() * dof_vel_scale;

			obs(37 + i) = (low_state->motor_state()[15+i].q() - default_angles[15+i]) * dof_pos_scale;
			obs(58 + i) = low_state->motor_state()[15+i].dq() * dof_vel_scale;  //61
		}

		std::cout << 0 << std::endl;
		//40
		for (int i = 0; i < 4; i++) //4个右手关节
		{
			dof_pos[17+i] = (low_state->motor_state()[22+i].q() - default_angles[22+i]) * dof_pos_scale;//dof_pos是21个关节的，但是motor_state包含23关节
			dof_vel[17+i] = low_state->motor_state()[22+i].dq() * dof_vel_scale;

			obs(41 + i) = (low_state->motor_state()[22+i].q() - default_angles[22+i]) * dof_pos_scale;  //44
			obs(62 + i) = low_state->motor_state()[22+i].dq() * dof_vel_scale;
		}
        //////////65
        // === 在这里插入打印 dof_pos 的代码 ===
        std::cout << "dof_pos: [";
        for(size_t i = 0; i < dof_pos.size(); ++i) {
            std::cout << dof_pos[i] << (i == dof_pos.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

		//历史观测
		for ( int i = 0; i < 84; ++i) {  //
			obs(66+i) = action_buf[i];
		}
		//149
		for ( int i = 0; i < 12; ++i) {  //
			obs(150+i) = ang_vel_buf[i];
		}
		std::cout << 12 << std::endl;
		//161
		for ( int i = 0; i < 84; ++i) {  //
			obs(162+i) = dof_pos_buf[i];
		}
		//245
		for ( int i = 0; i < 84; ++i) {  //
			obs(246+i) = dof_vel_buf[i];
		}
		std::cout << 11 << std::endl;
		//329
		for ( int i = 0; i < 12; ++i) {  //
			obs(330+i) = proj_g_buf[i];
		}
		//341
		for ( int i = 0; i < 4; ++i) {  //
			obs(342+i) = ref_motion_phase_buf[i];
		}
		//345
		for (int i = 0; i < 3; i++)
		{
			
			obs(346+i) = -R(2, i);
			gravity_orientation[i] = -R(2, i); // 记录重力方向
		}
		std::cout << 9 << std::endl;
		//348
		std::cout << 3 << std::endl;

		step_count += .02;
		float ref_motion_phase= step_count/period ; 
		ref_motion_phase = std::max(0.0f, std::min(ref_motion_phase, 1.0f)); //限制在0-1
		obs(349) = ref_motion_phase;
		std::cout << 4 << std::endl;
		//更新历史观测
		std::vector<float> act_vec(act.data(), act.data() + act.size());
		update_buffer(action_buf, act_vec, 21);
		update_buffer(ang_vel_buf, ang_vel, 3);
		update_buffer(dof_pos_buf, dof_pos, 21);
		update_buffer(dof_vel_buf, dof_vel, 21);
		update_buffer(proj_g_buf, gravity_orientation, 3);
		update_buffer(ref_motion_phase_buf, {std::min(ref_motion_phase, 1.0f)}, 1);
		std::cout << 5 << std::endl;

		// torch::Tensor torch_tensor = torch::from_blob(obs.data(), {1, obs.size()}, torch::kFloat).clone();		
   		// std::vector<float> input_tensor_values(torch_tensor.data_ptr<float>(), torch_tensor.data_ptr<float>() + torch_tensor.numel());
		// // 创建 ONNX 输入张量
		// std::array<int64_t, 2> input_shape{1, obs.size()};
		// Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		// 	allocator, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

		// // 执行推理
		// auto output_tensors = session.Run(Ort::RunOptions{nullptr},
		// 								&input_name, &input_tensor, 1,
		// 								&output_name, 1);

		// float* output_data = output_tensors[0].GetTensorMutableData<float>();
		// std::memcpy(act.data(), output_data, act.size() * sizeof(float));

		// ONNX Runtime inference
		Ort::AllocatorWithDefaultOptions allocator;
		std::vector<const char*> input_names = {"actor_obs"};
		std::vector<const char*> output_names = {"action"}; // 根据你的模型实际命名修改
		std::cout << 6 << std::endl;
		std::vector<int64_t> input_shape = {1, obs.size()};
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
			memory_info, obs.data(), obs.size(),
			input_shape.data(), input_shape.size()
		);
		std::cout << 7 << std::endl;
	
		// 推理
		auto output_tensors = session->Run(
			  
			Ort::RunOptions{nullptr},
			input_names.data(), &input_tensor, 1,
			output_names.data(), 1
		);
	
		// std::cout << 8 << std::endl;
		// 获取输出
		float* output_data = output_tensors[0].GetTensorMutableData<float>();
		std::cout << 9 << std::endl;
		std::memcpy(act.data(), output_data, act.size() * sizeof(float));
		std::cout << 10 << std::endl;

		//python robominic 参考程序
        // # update motion phase
        // self.counter_step += 1
        // motion_time = self.counter_step * 0.02
        // self.ref_motion_phase = motion_time / self.motion_length
        // motion_time = min(motion_time, self.motion_length)
        // print(progress_bar(motion_time, self.motion_length), end="", flush=True)



		//原本观测
		// for (int i = 0; i < 3; i++)
		// {
		// 	obs(i) = ang_vel_scale * low_state->imu_state().gyroscope()[i];
		// 	obs(i + 3) = -R(2, i);
		// }

		// if (obs(5) > 0)
		// {
		// 	break;
		// }

		// obs(6) = joy.ly * max_cmd[0] * cmd_scale[0];
		// obs(7) = joy.lx * -1 * max_cmd[1] * cmd_scale[1];
		// obs(8) = joy.rx * -1 * max_cmd[2] * cmd_scale[2];

		// // if (flag==0)
		// // {
		// for (int i = 0; i < 12; i++)
		// {
		// 	obs(9 + i) = (low_state->motor_state()[i].q() - default_angles[i]) * dof_pos_scale;
		// 	obs(21 + i) = low_state->motor_state()[i].dq() * dof_vel_scale;
		// }

		// }
		// else 
		// {
		// 	if (flag>8) flag=1;
		// 	for (int i = 0; i < 12; i++)
		// 	{
		// 		obs(9 + i) = (act(i)- default_angles[i]) * dof_pos_scale;
				
		// 	}
		// }
		// flag++;
		
		// obs.segment(33, 12) = act;

		// float phase = std::fmod(time / period, 1);
		// time += .02;
		// obs(45) = std::sin(2 * M_PI * phase);
		// obs(46) = std::cos(2 * M_PI * phase);
		
		
		// policy forward pt
		// torch::Tensor torch_tensor = torch::from_blob(obs.data(), {1, obs.size()}, torch::kFloat).clone();
		// std::vector<torch::jit::IValue> inputs;
		// inputs.push_back(torch_tensor);
		// torch::Tensor output_tensor = module.forward(inputs).toTensor();
		// std::memcpy(act.data(), output_tensor.data_ptr<float>(), output_tensor.size(1) * sizeof(float));


		// // ONNX Runtime inference
		// std::vector<const char*> input_names = {"actor_obs"};
		// std::vector<const char*> output_names = {"action"}; // 根据你的模型实际命名修改

		// std::vector<int64_t> input_shape = {1, obs.size()};
		// Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		// 	allocator, obs.data(), obs.size(), input_shape.data(), input_shape.size()
		// );

		// // 推理
		// auto output_tensors = session.Run(
		// 	Ort::RunOptions{nullptr},
		// 	input_names.data(), &input_tensor, 1,
		// 	output_names.data(), 1
		// );

		// // 获取输出
		// float* output_data = output_tensors[0].GetTensorMutableData<float>();
		// std::memcpy(act.data(), output_data, act.size() * sizeof(float));


		// std::cout << act.size() << std::endl;
		// std::cerr << "开始记录 act_output.csv 文件" << std::endl;
		// std::ofstream outfile("act_output.csv", std::ios::app); // 追加写入模式
		// if (outfile.is_open()) {
		// 	outfile << act[9] << "\n";  // 每次写入一行
		// 	outfile.close();
		// } else {
		// 	std::cerr << "无法打开 act_output.csv 文件" << std::endl;
		// }
// === 插入动作有效性检查（放在这里） ===
        

		std::ofstream outfile("act_log.csv", std::ios::app); // 以追加模式打开
		if (outfile.is_open()) {
			for (int i = 0; i < act.size(); ++i) {
				outfile << act(i);
				if (i < act.size() - 1) {
					outfile << ","; // 用逗号分隔
				}
			}
			outfile << "\n"; // 每帧数据写完后换行
			outfile.close();
		} else {
			std::cerr << "无法打开 act_log.csv 文件" << std::endl;
		}
	std::cerr << "打印 act_log.csv 文件" << std::endl;

		static Eigen::VectorXf last_act = act;
		bool act_valid = true;
		for (int i = 0; i < act.size(); ++i) {
			float v = act[i];
			if (!std::isfinite(v)) { act_valid = false; break; }
			// 预测映射到关节
			if (i < (int)dof_lower_limits.size()) {
				float q = default_angles[i] + v * action_scale;
				if (q < dof_lower_limits[i] - 1e-4f || q > dof_upper_limits[i] + 1e-4f) {
					std::cerr << "[WARN] joint " << i << " predicted q=" << q
							<< " out of [" << dof_lower_limits[i] << ","
							<< dof_upper_limits[i] << "]\n";
					act_valid = false;
					break;
				}
			}
		}
		if (!act_valid) {
			act = last_act;
		} else {
			last_act = act;
		}
        
		std::cout << 11 << std::endl;
		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();
		// leg  // 12个腿部关节+1个腰部关节
		for (int i = 0; i < 13; i++)
		{
			low_cmd->motor_cmd()[i].q() = act(i) * action_scale + default_angles[i];
			low_cmd->motor_cmd()[i].kp() = kps[i];
			low_cmd->motor_cmd()[i].kd() = kds[i];
			low_cmd->motor_cmd()[i].dq() = 0;
			low_cmd->motor_cmd()[i].tau() = 0;
		}
		//+4个手部关节
		for (int i = 0; i < 4; i++)
		{
			low_cmd->motor_cmd()[15+i].q() = act(13+i) * action_scale + default_angles[15+i];
			low_cmd->motor_cmd()[15+i].kp() = kps[13+i];
			low_cmd->motor_cmd()[15+i].kd() = kds[13+i];
			low_cmd->motor_cmd()[15+i].dq() = 0;
			low_cmd->motor_cmd()[15+i].tau() = 0;
		}

		//4个手部关节
		for (int i = 0; i < 4; i++)
		{
			low_cmd->motor_cmd()[22+i].q() = act(17+i) * action_scale + default_angles[22+i];
			low_cmd->motor_cmd()[22+i].kp() = kps[17+i];
			low_cmd->motor_cmd()[22+i].kd() = kds[17+i];
			low_cmd->motor_cmd()[22+i].dq() = 0;
			low_cmd->motor_cmd()[22+i].tau() = 0;
		}

		//固定关节部分
		for (int i = 0; i < 8; i++)
		{
			int motor_idx = static_cast<int>(fixed_joint2motor_idx[i]); 
			low_cmd->motor_cmd()[motor_idx].q() =  fixed_target_angles[i];
			low_cmd->motor_cmd()[motor_idx].kp() = fixed_kps[i];
			low_cmd->motor_cmd()[motor_idx].kd() = fixed_kds[i];
			low_cmd->motor_cmd()[motor_idx].dq() = 0;
			low_cmd->motor_cmd()[motor_idx].tau() = 0;
		}

		// //左手手腕   需要修改
		// low_cmd->motor_cmd()[19].q() =  default_angles[19];
		// low_cmd->motor_cmd()[19].kp() = kps[19];
		// low_cmd->motor_cmd()[19].kd() = kds[19];
		// low_cmd->motor_cmd()[19].dq() = 0;
		// low_cmd->motor_cmd()[19].tau() = 0;
		// //右手手腕
		// low_cmd->motor_cmd()[26].q() =  default_angles[26];
		// low_cmd->motor_cmd()[26].kp() = kps[26];
		// low_cmd->motor_cmd()[26].kd() = kds[26];
		// low_cmd->motor_cmd()[26].dq() = 0;
		// low_cmd->motor_cmd()[26].tau() = 0;

		// // waist armg
		// for (int i = 12; i < 29; i++)
		// {
		// 	low_cmd->motor_cmd()[i].q() = arm_waist_target[i - 12];
		// 	low_cmd->motor_cmd()[i].kp() = arm_waist_kps[i - 12];
		// 	low_cmd->motor_cmd()[i].kd() = arm_waist_kds[i - 12];
		// 	low_cmd->motor_cmd()[i].dq() = 0;
		// 	low_cmd->motor_cmd()[i].tau() = 0;
		// }
		std::cout << 12 << std::endl;

        if (act_valid) {
            mLowCmdBuf.SetDataPtr(low_cmd);
        } else {
            // 可选：打印一条信息，表明由于数据无效，本帧未发送新指令
            std::cerr << "[WARN] Skipping command send due to invalid 'act' data." << std::endl;
        }
        if(step_count>1.0)
		{
		mLowCmdBuf.SetDataPtr(low_cmd);
		}


		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::damp()
{
	std::cout << "damping\n";
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	while (true)
	{
		auto low_cmd = std::make_shared<unitree_hg::msg::dds_::LowCmd_>();
		for (auto &cmd : low_cmd->motor_cmd())
		{
			cmd.kp() = 0;
			cmd.kd() = 8;
			cmd.dq() = 0;
			cmd.tau() = 0;
		}

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}


void Controller::low_state_message_handler(const void *message)
{
	unitree_hg::msg::dds_::LowState_* ptr = (unitree_hg::msg::dds_::LowState_*)message;
	mLowStateBuf.SetData(*ptr);
	std::memcpy(&joy, ptr->wireless_remote().data(), ptr->wireless_remote().size() * sizeof(uint8_t));
}

void Controller::low_cmd_write_handler()
{
	if (auto lowCmdPtr = mLowCmdBuf.GetDataPtr())
	{
		lowCmdPtr->mode_machine() = mLowStateBuf.GetDataPtr()->mode_machine();
		lowCmdPtr->mode_pr() = 0;
		for (auto &cmd : lowCmdPtr->motor_cmd())
		{
			cmd.mode() = 1;
		}
		lowCmdPtr->crc() = crc32_core((uint32_t*)(lowCmdPtr.get()), (sizeof(unitree_hg::msg::dds_::LowCmd_) >> 2) - 1);
		lowcmd_publisher->Write(*lowCmdPtr);
	}
}
