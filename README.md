## Flash-RAG

Flash-RAG 是一个基于 **vLLM** 的高并发垂直领域智能问答引擎，当前主要聚焦于 **法律条文咨询助手** 场景。

---

## 项目结构

```text
Flash-RAG/
├── config/                # 配置文件目录
│   └── train_config.yaml  # 训练与模型相关的全部参数
├── datasets/                  # 训练/评测数据
│   └── train.jsonl
├── output/                # 训练输出与日志（自动生成）
├── train.py               # 训练脚本，只负责逻辑，不写死参数
└── requirements.txt       # 项目依赖
```

---

## 快速开始

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 准备数据（确保 `data/train.jsonl` 存在且格式正确）。

3. 启动训练：

```bash
python train.py
```

根据需要修改 `config/train_config.yaml` 即可调整模型、数据和训练参数。

