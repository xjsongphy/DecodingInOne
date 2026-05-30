# exp 实验目录

## 1) 训练模型（你当前要的）

脚本：`experiments/ising/ising_train_model.py`

用途：提取 Ising 风格的**训练实验主链路**（仅训练，不做完整解码流程），包含：

1. 配置加载（YAML + CLI 覆盖）
2. Stim 生成表面码 memory 电路
3. 采样训练/验证集（detectors -> observables）
4. PyTorch 训练（MLP）
5. 输出 checkpoint 和训练报告

运行：

```bash
uv run python experiments/ising/ising_train_model.py --config experiments/ising/ising_train.yaml
```

快速 smoke：

```bash
uv run python experiments/ising/ising_train_model.py \
  --train-shots 4000 --val-shots 1000 --epochs 2 --batch-size 256 --out-dir experiments/ising/train_smoke
```

输出：

- `experiments/ising/train_output/best_model.pt`
- `experiments/ising/train_output/train_report.json`

## 2) 推理流程（之前提取的）

脚本：`experiments/ising/ising_full_pipeline.py`

说明：这是端到端推理/评估流程（含 PyMatching baseline 与延迟统计），不是训练脚本。

运行：

```bash
uv run python experiments/ising/ising_full_pipeline.py --config experiments/ising/ising_pipeline.yaml
```
