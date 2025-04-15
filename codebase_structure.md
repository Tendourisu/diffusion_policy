# Diffusion Policy 代码结构分析

## 整体架构

Diffusion Policy是一个基于扩散模型的机器人策略学习框架，其代码结构设计遵循了模块化和可扩展的原则。代码库的主要结构围绕以下核心组件构建：

## 关键组件及其关系

### 1. Workspace（工作区）

Workspace是整个实验流程的核心封装，负责管理训练和评估的完整生命周期。

- 位于`diffusion_policy/workspace/`目录
- 所有Workspace继承自`BaseWorkspace`
- 每个具体实现（如`train_diffusion_unet_image_workspace.py`）对应一种特定的训练方法
- 通过`run`方法管理整个实验流程
- 负责模型检查点的保存与加载

### 2. Policy（策略）

Policy实现了模型的推理和部分训练逻辑。

- 位于`diffusion_policy/policy/`目录
- 基类分为`BaseLowdimPolicy`和`BaseImagePolicy`，分别对应低维和图像观察空间
- 每个Policy通常与对应的Workspace配对
- 实现了`predict_action`方法来根据观察预测动作
- 实现了`set_normalizer`方法处理观察和动作的归一化
- 可能包含`compute_loss`方法用于训练

### 3. Dataset（数据集）

Dataset处理训练数据的加载和预处理。

- 位于`diffusion_policy/dataset/`目录
- 继承自`torch.utils.data.Dataset`
- 返回符合接口规范的样本
- 提供`get_normalizer`方法返回用于归一化的`LinearNormalizer`对象
- 通常结合`ReplayBuffer`和`SequenceSampler`来生成训练样本

### 4. EnvRunner（环境运行器）

EnvRunner抽象了不同任务环境之间的细微差别。

- 位于`diffusion_policy/env_runner/`目录
- 实现了`run`方法，接受一个`Policy`对象进行评估，并返回日志和指标
- 通常使用`gym.vector.AsyncVectorEnv`的修改版本来并行化环境运行，提高评估速度

### 5. 配置文件

配置文件用于定义实验所需的所有参数和组件设置。

- 位于`diffusion_policy/config/`目录
- 任务配置在`config/task/`子目录下
- 工作区配置直接在`config/`目录下
- 使用Hydra框架管理配置

### 6. 模型组件

模型定义和实现分布在多个子目录中：

- `diffusion_policy/model/diffusion/`: 扩散模型相关组件
- `diffusion_policy/model/common/`: 通用模型组件，如归一化器
- `diffusion_policy/model/vision/`: 视觉模型组件
- `diffusion_policy/model/bet/`: BET（Behavior Transformer）模型实现

### 7. 辅助组件

#### ReplayBuffer

- 位于`diffusion_policy/common/replay_buffer.py`
- 用于存储演示数据集的关键数据结构
- 支持内存和磁盘存储，带有分块和压缩功能
- 使用`zarr`格式进行存储

#### SharedMemoryRingBuffer

- 位于`diffusion_policy/shared_memory/shared_memory_ring_buffer.py`
- 无锁FILO（先进后出）数据结构
- 在实际机器人实现中广泛使用，以避免多进程通信中的序列化和锁定开销

### 8. 接口规范

代码库定义了清晰的接口规范，使得不同组件能够无缝协作：

#### 低维接口
- `LowdimPolicy`: 接受形状为`(B,To,Do)`的观察张量
- `LowdimDataset`: 返回形状为`(To,Do)`的观察张量和形状为`(Ta,Da)`的动作张量
- 使用`LinearNormalizer`进行归一化

#### 图像接口
- `ImagePolicy`: 接受包含多个键的观察字典，其中可能包含图像数据
- `ImageDataset`: 返回观察字典和动作张量
- 同样使用`LinearNormalizer`进行归一化

## 组件之间的工作流程

1. `train.py`作为入口点，使用Hydra加载配置
2. 配置创建一个`Workspace`实例
3. `Workspace`加载数据集，创建`Policy`和优化器
4. 训练循环中，`Workspace`从`Dataset`获取批次数据
5. `Policy`计算损失，优化器更新参数
6. 定期使用`EnvRunner`评估当前策略
7. 保存检查点和指标

## 扩展框架

### 添加新任务
- 创建新的`Dataset`类
- 创建新的`EnvRunner`类
- 在`config/task/`中添加配置文件

### 添加新方法
- 创建新的`Policy`类
- 创建对应的`Workspace`类
- 添加相应的配置文件

## 设计理念

该代码库的结构设计遵循以下原则：
1. 实现N个任务和M个方法只需要O(N+M)量的代码，而不是O(N*M)
2. 保持最大的灵活性
3. 任务和方法的实现相互独立
4. 通过简单统一的接口连接任务和方法

这种设计选择可能导致任务和方法之间的代码重复，但带来了能够在不影响其余部分的情况下添加/修改任务/方法的好处，并且能够通过线性阅读代码来理解任务/方法。 

## 遇到的问题

```sh
pip uninstall -y huggingface_hub                    
pip install huggingface_hub==0.11.1
```
- 在安装`huggingface_hub`时，可能会遇到版本不兼容的问题。建议使用`0.11.1`版本。

```sh
pip uninstall -y torch torchvision
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117  
```
- 在安装`torch`和`torchvision`时，可能会遇到cuda版本不兼容的问题。建议使用`1.13.1`和`0.14.1`版本，并指定CUDA版本为`cu117`。 