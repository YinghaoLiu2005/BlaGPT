# synthetic_tests_v2

独立可迁移的 KLA 三任务实验包：直接复用 `kla-anonymous` 的 MQAR、Palindrome、64-Stack
数据生成器和训练脚本，替换为 **KDA、GDN、Mamba2、Mamba2-GQA、Mamba2-GQA-Sink**。
复制整个目录即可使用，不需要原来的 BlaGPT 或 KLA 仓库。

## 配置基准

以 `upstream/kla/run_*_experiment.sh` 的**实际默认参数**为准，连同训练脚本的未显式覆盖项一起保留。
不是旧 `synthesis_test/kla_v3` 的重新实现，也没有继承其 MQAR 数据修改、学习率搜索或重复种子实验。

| 参数 | MQAR | Palindrome | 64-Stack |
|---|---:|---:|---:|
| 训练 seq_len | 256 | 1024（source 长度） | 1024 |
| 实际模型输入长度 | 256 | 2049 | 1024 |
| 最终评估长度 | 256/512/1024/2048 | source 256/512/1024/2048 | 256/512/1024/2048 |
| 最大训练步数 | 10000 | 20000 | 20000 |
| batch size | 32 | 32 | 32 |
| 学习率 | 0.001 | 0.001 | 0.001 |
| 验证间隔 | 200 | 500 | 500 |
| 保存间隔 | 1000 | 1000 | 1000 |
| early-stop patience | 10 | 20 | 20 |
| 层数 / hidden size | 2 / 256 | 2 / 256 | 2 / 256 |
| 头数 / head dim | **4 / 64** | 2 / 128 | 2 / 128 |
| 词表 | 8192 | 128 | 128 |

三个任务均为 train/val/test = 20000/2000/2000，seed=42。原始 AdamW 对全部参数使用
weight_decay=0.1、betas=(0.9,0.95)，梯度裁剪 1.0，GPU bf16-mixed，无 scheduler。
`_no_weight_decay` 属性不会改变原始优化器的行为，因此 Sink 参数在本实验也遵循相同 weight decay。
训练保留原始 early stopping，实际步数可能少于上限。

MQAR 保留 key_len=1、num_pairs=32、power_law_alpha=0.1、原始有放回 key 采样和查询布局。
Palindrome 启用 `predict_first_token`，最大评估输入为 **4097 tokens**。
Stack 保留 64 个栈、26 种值、push_prob=0.6。
Stack 的原有可选 `ENABLE_LR_SWEEP=1` 仍然可用，默认不开启。

## 五个架构与兼容层

五个模型共用 KLA GatedDeltaNet 任务 preset 的外壳：两层 sequential pre-norm residual，
RMSNorm eps=1e-5，SwiGLU intermediate_size=1024，embedding/output 权重绑定，
保留 KLA `mamba_init=True` 的初始化规则。

- KDA/GDN：冻结的 FLA 0.5.0 实现，expand_v=1，conv_size=4，allow_neg_eigval=False。
- Mamba2：冻结的本仓库实现，expand=1，d_state=128，ngroups=1，d_conv=4，chunk_size=256。
- Mamba2-GQA：同样的 SSM 设置，窗口 512，KV heads=1，RoPE theta=500000，dropout=0。
- Mamba2-GQA-Sink：独立 `mamba-gqa-sink.py`，显式 `use_attn_sink=True`，初值 -2，
  每层每个 query head 一个可训练 sink。依赖 FlashAttention **2.6.3** 的内部 API。

MQAR 的 4 头是 KLA 原配置，不统一改成旧实验的 2 头。所有 Mamba 模型都使用上述共同外壳，
不使用 KLA 原生 Mamba-only 无 FFN 外壳；这属于本实验替换的模型架构。

为去除 xformers、独立 dropout_layer_norm 扩展以及 FLA 环境对 flash-attn loss 的依赖，
`lit_gpt/` 使用同公式的 PyTorch RMSNorm、SwiGLU 和 mean cross entropy（ignore_index=-100）。
这是算子实现的兼容调整；不承诺与旧 fused CUDA kernel 逐位一致。
三个 `*_data.py`、三个 `train_*.py` 和 `lit_gpt/config.py` 均与原文件逐字节一致。
模型注册表在导入时添加五架构名称。分析脚本原样保留，绘图样式增加五架构名称和颜色。

## 在其他服务器安装

需要 NVIDIA CUDA GPU；建议每次用一张卡串行运行，先用下面的验证命令检查显存和内核兼容性。
本机验证使用两套环境，避免为 Sink 的 FlashAttention 2.6.3 与新 FLA 强行共用同一套 PyTorch。
`vendor/` 已包含实际使用的 FLA 和定制 Mamba Python 源码；CUDA 二进制扩展需在目标服务器安装。

```bash
cd synthetic_tests_v2
# Python 3.10/3.11；Mamba 环境安装扩展需要匹配的 CUDA toolkit/nvcc。
PYTHON=python3.10 bash setup_env.sh fla
PYTHON=python3.10 bash setup_env.sh mamba

export FLA_PYTHON="$PWD/.venvs/fla/bin/python"
export MAMBA_PYTHON="$PWD/.venvs/mamba/bin/python"
export CUDA_VISIBLE_DEVICES=0
```

安装脚本分别使用 PyTorch 2.7.1/cu118 和 2.5.0/cu124。已有匹配环境时，只需设置两个解释器路径，
再安装对应 `requirements-*.txt`，无需重新创建环境。不要把系统安装的原版 Mamba 当作 GQA/Sink 源码；
模型适配器会优先加载随包冻结的 `vendor/mamba`，测试会检查实际导入路径。
FLA 内核使用随包 `vendor/fla`，不需要单独安装 `flash-linear-attention`。

## 验证和运行

```bash
# 只打印正式命令，不生成数据或开始训练。
DRY_RUN=1 bash run_all.sh

# CPU：源码哈希、启动参数一致性、数据一致性、独立任务 oracle、15 个模型配置。
"$MAMBA_PYTHON" verify.py --cpu-only

# 完整 GPU 测试；测试数据、checkpoint、日志单独放入 test_outputs。
"$MAMBA_PYTHON" verify.py

# 正式实验：三个任务 × 五个模型，共 15 个默认 run，按顺序运行。
bash run_all.sh

# 也可分别运行单个任务。
bash run_mqar_experiment.sh
bash run_palindrome_experiment.sh
bash run_stack_experiment.sh
```

完整验证包括每个模型/任务在正式训练长度上的前反向、全部梯度有限、参数更新、未来 token 不影响前缀、
checkpoint 重载逐位一致、最大评估长度前向，以及**原训练脚本**的 2 步训练→验证→保存→加载→测试→长度外推。
还会执行 `check_operators.py`，对照 KLA 原初始化函数、原 RMSNorm、FlashAttention 交叉熵，
以及独立 dense softmax 的 Sink 输出和 Q/K/V/sink 梯度。
短训练用于验证流程，不代表正式训练收敛或最终准确率。首次 Triton 编译可能需要数分钟。
默认验证输出目录带时间戳；显式传 `--output` 时应使用新目录，以避免旧结果混入。

训练结果位于 `out/<task>/<experiment>/results.json`，数据位于 `data/<task>/`。
保留原脚本的行为：存在 `results.json` 时跳过该 run；已有 checkpoint 的未完成 run **不会自动断点续训**。
默认 W&B 使用 offline 本地日志，不要求登录；如需在线记录可自行设置 `WANDB_MODE=online`。

```bash
"$MAMBA_PYTHON" mqar_analysis.py --out_root out/mqar --save_dir out/analysis/mqar
"$MAMBA_PYTHON" palindrome_analysis.py --out_root out/palindrome --save_dir out/analysis/palindrome \
  --include_experiment '' --exclude_path_keywords '_backup_'
"$MAMBA_PYTHON" stack_analysis.py --out_root out/stack --save_dir out/analysis/stack
# 或直接运行上述三个命令：
bash analyze_all.sh
```

Palindrome 原分析脚本默认筛掉 `pf1` 并限定旧实验目录名，与其训练 launcher 的默认设置不一致。
这里通过上面的 CLI 参数解除旧筛选，保留原分析代码不变。
MQAR 原结果 JSON 不含训练曲线，原分析脚本的收敛曲线依赖 W&B `run-*` 日志；
默认 offline 模式下可直接使用最终评估表和长度外推图。

## 迁移与溯源

```bash
bash pack.sh
# 把 synthetic_tests_v2.tar.gz 复制到目标服务器，解压后按上面的步骤安装和运行。
tar -xzf synthetic_tests_v2.tar.gz
```

打包排除正式数据、结果、测试临时文件、虚拟环境和 Python 缓存。`source_manifest.json` 记录
KLA commit 和原始文件/冻结依赖的 SHA-256；`upstream/kla/` 保存原文件及许可证，便于独立检查差异。
`TEST_REPORT.json` 保存交付时的验证结果。每台新服务器都应重新运行 `verify.py`。
