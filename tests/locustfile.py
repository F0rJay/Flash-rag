"""
Locust 压测脚本：测试 LegalFlash-RAG API 性能
使用方法：
    locust -f tests/locustfile.py --host=http://localhost:8080
然后在浏览器打开 http://localhost:8089 进行压测
"""
from locust import HttpUser, task, between
import json
import random


# 测试问题列表（法律相关）
TEST_QUERIES = [
    "什么是合同违约？",
    "如何申请劳动仲裁？",
    "离婚财产如何分割？",
    "交通事故责任如何认定？",
    "如何申请强制执行？",
    "什么是正当防卫？",
    "如何申请法律援助？",
    "合同无效的情形有哪些？",
    "如何计算违约金？",
    "什么是不可抗力？",
    "如何申请行政复议？",
    "什么是诉讼时效？",
    "如何申请财产保全？",
    "什么是格式条款？",
    "如何申请执行异议？",
]


class LegalFlashRAGUser(HttpUser):
    """LegalFlash-RAG API 压测用户"""
    
    wait_time = between(1, 3)  # 用户请求间隔 1-3 秒
    
    def on_start(self):
        """用户启动时执行（可选）"""
        # 可以在这里进行登录等操作
        pass
    
    @task(3)
    def test_chat_endpoint(self):
        """测试聊天接口（权重 3）"""
        query = random.choice(TEST_QUERIES)
        payload = {
            "query": query,
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False
        }
        
        with self.client.post(
            "/api/rag/chat",
            json=payload,
            catch_response=True,
            name="POST /api/rag/chat"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    response.success()
                else:
                    response.failure(f"响应格式错误: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def test_chat_stream_endpoint(self):
        """测试流式聊天接口（权重 1）"""
        query = random.choice(TEST_QUERIES)
        payload = {
            "query": query,
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": True
        }
        
        with self.client.post(
            "/api/rag/chat",
            json=payload,
            catch_response=True,
            stream=True,
            name="POST /api/rag/chat (stream)"
        ) as response:
            if response.status_code == 200:
                # 检查流式响应
                chunks_received = 0
                for chunk in response.iter_lines():
                    if chunk:
                        chunks_received += 1
                        if chunks_received > 0:
                            break  # 至少收到一个块就认为成功
                
                if chunks_received > 0:
                    response.success()
                else:
                    response.failure("未收到流式数据")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(5)
    def test_health_endpoint(self):
        """测试健康检查接口（权重 5，高频）"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="GET /health"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") in ["healthy", "degraded"]:
                    response.success()
                else:
                    response.failure(f"服务不健康: {data.get('status')}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def test_metrics_endpoint(self):
        """测试监控指标接口（权重 2）"""
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="GET /metrics"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "latency" in data and "throughput" in data:
                    response.success()
                else:
                    response.failure("监控指标格式错误")
            else:
                response.failure(f"HTTP {response.status_code}")


# 压测配置类（可选）
class LegalFlashRAGStressTest(HttpUser):
    """压力测试：高并发场景"""
    
    wait_time = between(0.1, 0.5)  # 更短的等待时间，模拟高并发
    
    @task
    def stress_test_chat(self):
        """高并发聊天测试"""
        query = random.choice(TEST_QUERIES)
        payload = {
            "query": query,
            "temperature": 0.1,
            "max_tokens": 256,  # 较短的响应，减少延迟
            "stream": False
        }
        
        self.client.post("/api/rag/chat", json=payload, name="Stress Test Chat")

