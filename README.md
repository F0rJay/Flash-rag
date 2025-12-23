# Flash-rag
Flash-RAG：基于 vLLM 的高并发垂直领域智能问答引擎
法律条文咨询助手
Flash-RAG/
├── config/                # [新建] 存放配置文件的文件夹
│   └── train_config.yaml  # [新建] 所有的参数都在这里
├── data/                  # 存放数据
│   └── train.jsonl
├── output/                # [自动生成] 存放训练结果和日志
├── train.py               # [修改] 只负责逻辑，不包含硬编码参数
└── requirements.txt       # 项目依赖
