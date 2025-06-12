# OWL: Optimized Workforce Learning for General Multi-Agent Assistance for Real-World Task Automation

We present Workforce, a hierarchical multi-agent framework that decouples planning from execution through a modular
architecture with a domain-agnostic Planner, Coordinator, and specialized Workers. This enables cross-domain transfer by
allowing worker modification without full system retraining. On the GAIA benchmark, Workforce achieves state-of-the-art
69.70% accuracy, outperforming commercial systems.

This repository contains inference part code for the OWL framework (Workforce).

## Inference

The framework is based on `camel-0.2.46` version with minor modifications. To reproduce Workforce inference performance on GAIA benchmark (69.70% - Claude-3.7 accuracy on GAIA benchmark, pass@1, and 60.61% - GPT-4o accuracy on GAIA benchmark, pass@3), follow the steps below:

### Installation and Setup

1. Create a Python 3.11 Conda environment:

```bash
conda create -n owl python=3.11
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up envionment variables:

copy `.env.example` to `.env` and set the environment variables, and set the keys in `.env` file.

4. Run the inference:

- For reproducing results using GPT-4o, run:

```bash
python run_gaia_workforce.py
```

- For reproducing results using Claude-3.7, run:

```bash
python run_gaia_workforce_claude.py
```

You can modify `test_idx` variable to specify the test case.



==================================
## planning准确率比较(关掉browser_use)
- gpt-4o: 26/53
- qwen3-4b: 21/53
- qwen2.5-3b-instruct: 17/53
- qwen2.5-3b-instruct(合成数据训练后,zyq): 23/53
- qwen2.5-3b-instruct(合成数据训练后,zyx): 23/53


## 使用vllm测试本地模型
1. 在有gpu的开发机上用vllm启动模型，e.g. CUDA_VISIBLE_DEVICES=0,1 vllm serve /mnt/public/algm/models/Qwen3-4B --enable-auto-tool-choice --tool-call-parser hermes --tensor-parallel-size 2 --gpu-memory-utilization 0.5 --port 8001
2. 将开发机的vllm端口，例如这里是8001映射到本机或者一台可以被公网访问的机器上：
    * 映射到公网访问的机器:
        ```shell
        autossh -M 0 -R 39929:127.0.0.1:8001 \
        -o "ExitOnForwardFailure=yes" \
        -o "ServerAliveInterval=10" \
        -o "ServerAliveCountMax=3" \
        root@59.110.169.144
        ```
    * 映射到本机：
        ```shell
        ssh -L 39929:localhost:8001 -p 40675 yzy@111.51.90.14
        ```
3.修改run_gaia_workforce.py中planner模型配置
```python
    task_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.VLLM,
            model_type="/mnt/public/algm/zhuangyueqing/public_logs/qwen_sft/Qwen2.5-3B-Instruct__question_v1_1000_decompose_train_jsonl/final",
            model_config_dict=model_config_dict,
            # url="http://59.110.169.144:39929/v1",
            url="http://127.0.0.1:39929/v1",
        )
    }
```