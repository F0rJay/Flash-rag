from typing import Any, List, Mapping, Optional, Iterator
import requests
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field
import json

# 这是 LangChain 框架中的高级工程模式：创建自定义 LLM
class CustomVLLM(BaseLLM):
    """自定义 LLM 类，用于连接正在运行的 vLLM API 服务."""
    
    # 从配置中获取 vLLM 服务的 URL
    api_url: str = Field(default="http://localhost:8000/v1/completions")
    
    @property
    def _llm_type(self) -> str:
        return "custom_vllm"

    # 核心方法：调用推理服务
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        
        # 1. 构造请求体 (Payload)
        payload = {
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.1,
            "stop": stop or [],
            "stream": False # 暂不启用流式输出，简化测试
        }

        # 2. 发送请求到 vLLM API
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() # 检查HTTP错误
        except requests.exceptions.ConnectionError:
            return "ERROR: Could not connect to vLLM server at http://localhost:8000. Is it running?"
        
        # 3. 解析 vLLM 返回的 JSON
        data = response.json()
        
        # 4. 提取生成的文本 (vLLM 返回格式)
        text = data["choices"][0]["text"]
        
        return text
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> LLMResult:
        """生成方法，返回 LLMResult 对象"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """用于日志记录和调试"""
        return {"api_url": self.api_url}
    
    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        流式生成方法，返回一个生成器，逐 token 输出
        
        Args:
            prompt: 输入提示词
            stop: 停止词列表
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Yields:
            str: 每个生成的 token 文本
        """
        # 构造请求体
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
            "stream": True  # 启用流式输出
        }
        
        try:
            # 发送流式请求
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            # 解析 SSE 流式响应
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    # SSE 格式: data: {...}
                    if line_text.startswith('data: '):
                        data_str = line_text[6:]  # 去掉 "data: " 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            # 提取生成的文本
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                text = delta.get('text', '')
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.ConnectionError:
            yield "ERROR: Could not connect to vLLM server. Is it running?"
        except Exception as e:
            yield f"ERROR: {str(e)}"
    
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        调用方法（兼容 LangChain）
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        return self._call(prompt, **kwargs)