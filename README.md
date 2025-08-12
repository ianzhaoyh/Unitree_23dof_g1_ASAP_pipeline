# ASAP-Unitree-23dof-pipeline

这是一个将 ASAP 框架应用于宇树 G1 (23自由度) 机器人的训练和部署流程。

---

## 步骤一：环境设置 🛠️

首先，我们需要配置 Conda 环境并安装 Isaac Gym。

### 1. 创建并激活 Conda 环境
```bash
conda create -n asap23 python=3.8
conda activate asap23
```

### 2. 安装 Isaac Gym
从 NVIDIA 官网下载 Isaac Gym Preview 4，然后解压并安装。
```bash
# 假设您已下载 isaac-gym-preview-4.tar.gz
tar -xvzf isaac-gym-preview-4.tar.gz
```
安装其 Python API:
```bash
pip install -e isaacgym/python
```
通过运行示例来测试 Isaac Gym 是否安装成功：
```bash
# 进入 isaacgym/python/examples 目录
cd isaacgym/python/examples
python 1080_balls_of_solitude.py
```

### 3. 安装项目依赖
回到项目根目录，安装 `HumanoidVerse` 和其他依赖项。
```bash
# 确保您在 ASAP23dof 根目录
pip install -e .
pip install -e isaac_utils
```

---

## 步骤二：在 Isaac Gym 中训练 🏃‍♂️

完成环境设置后，可以开始在 Isaac Gym 中进行训练。

### 运行训练测试
使用以下命令启动一个简单的训练任务，以验证所有组件是否正常工作。
```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=1 \
project_name=TestIsaacGymInstallation \
experiment_name=G123dof_loco \
headless=False
```

---

## 步骤三：Sim2Sim (MuJoCo 仿真) 🤖

将训练好的策略迁移到 MuJoCo 仿真环境中进行验证。

### 1. 安装 MuJoCo
```bash
pip install mujoco
```

### 2. 配置策略和模型路径
在运行仿真前，请务必在 `deploy_g1_23_21.py` 脚本中修改策略 (`.onnx` 文件) 和机器人模型 (`.xml` 文件) 的路径。
```python
# 示例: deploy_g1_23_21.py
# policy_path = "/path/to/your/trained/model.onnx"
# xml_path = "/path/to/your/g1_23dof.xml"
```

### 3. 运行 MuJoCo 仿真
```bash
# 切换到包含部署脚本的目录
cd deploy/deploy_mujoco 
python deploy_g1_23_21.py
```

---

## 步骤四：Sim2Real (部署到实体机器人) 🦾

将策略部署到宇树 G1 实体机器人上。

### 1. 安装 Unitree SDK
参考宇树官方文档，安装与机器人通信所需的 `unitree_sdk2`。

unitree_sdk2
```bash
mkdir build
cd build
cmake ..
make
```
再进入deploy/deploy_real/cpp_g1下
下载LibTorch,onnxruntime    `


```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cpu.zip

cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz
tar xf onnxruntime-linux-x64-1.17.3.tgz
```
To build the project, executable the following steps
```bash
mkdir build
cd build
cmake ..
make -j4
```
如果编译失败，关注修改makefilelist.txt里的内容 添加包的路径们

After successful compilation, executate the program with:
```bash
./g1_deploy_run {net_interface}
```
Replace {net_interface} with your actual network interface name (e.g., eth0, wlan0).
使用ifconfig从查询以太网接口名