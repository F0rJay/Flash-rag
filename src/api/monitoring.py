"""
监控模块：GPU 使用率、延迟、吞吐量统计
"""
import time
import asyncio
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
import psutil
import requests

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️  pynvml 未安装，GPU 监控功能将不可用。安装: pip install nvidia-ml-py3")


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, vllm_url: str = "http://localhost:8000", max_history: int = 1000):
        """
        初始化指标收集器
        
        Args:
            vllm_url: vLLM 服务地址
            max_history: 历史记录最大数量
        """
        self.vllm_url = vllm_url
        self.max_history = max_history
        
        # 请求延迟历史记录（秒）
        self.latency_history: deque = deque(maxlen=max_history)
        
        # 吞吐量统计（请求数/秒）
        self.request_timestamps: deque = deque(maxlen=max_history)
        
        # 总请求数
        self.total_requests = 0
        
        # 总错误数
        self.total_errors = 0
        
        # 初始化 GPU 监控
        self.gpu_available = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = True
                print(f"✅ GPU 监控已初始化，检测到 {self.gpu_count} 个 GPU")
            except Exception as e:
                print(f"⚠️  GPU 监控初始化失败: {e}")
        
        # 启动时间
        self.start_time = time.time()
    
    def record_request(self, latency: float, success: bool = True):
        """
        记录请求指标
        
        Args:
            latency: 请求延迟（秒）
            success: 是否成功
        """
        self.latency_history.append(latency)
        self.request_timestamps.append(time.time())
        self.total_requests += 1
        if not success:
            self.total_errors += 1
    
    def get_latency_stats(self) -> Dict:
        """获取延迟统计"""
        if not self.latency_history:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "count": 0
            }
        
        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)
        
        return {
            "avg": sum(sorted_latencies) / n,
            "min": min(sorted_latencies),
            "max": max(sorted_latencies),
            "p50": sorted_latencies[int(n * 0.5)] if n > 0 else 0.0,
            "p95": sorted_latencies[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0.0,
            "count": n
        }
    
    def get_throughput(self, window_seconds: int = 60) -> float:
        """
        获取吞吐量（请求数/秒）
        
        Args:
            window_seconds: 时间窗口（秒）
        """
        if not self.request_timestamps:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # 统计时间窗口内的请求数
        recent_requests = sum(1 for ts in self.request_timestamps if ts >= cutoff_time)
        
        return recent_requests / window_seconds if window_seconds > 0 else 0.0
    
    def get_gpu_metrics(self) -> List[Dict]:
        """
        获取 GPU 指标
        
        Returns:
            List[Dict]: 每个 GPU 的指标列表
        """
        if not self.gpu_available:
            return []
        
        gpu_metrics = []
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU 名称
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # 显存使用情况
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_total = mem_info.total / 1024**3  # GB
                mem_used = mem_info.used / 1024**3  # GB
                mem_free = mem_info.free / 1024**3  # GB
                mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                
                # GPU 利用率
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # 温度
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                # 功耗
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0  # W
                except:
                    power = None
                    power_limit = None
                
                gpu_metrics.append({
                    "index": i,
                    "name": name,
                    "memory": {
                        "total_gb": round(mem_total, 2),
                        "used_gb": round(mem_used, 2),
                        "free_gb": round(mem_free, 2),
                        "utilization_percent": round(mem_util, 2)
                    },
                    "utilization_percent": gpu_util,
                    "temperature_celsius": temp,
                    "power_watts": round(power, 2) if power else None,
                    "power_limit_watts": round(power_limit, 2) if power_limit else None
                })
        except Exception as e:
            print(f"⚠️  获取 GPU 指标失败: {e}")
        
        return gpu_metrics
    
    def get_cpu_metrics(self) -> Dict:
        """获取 CPU 指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            mem = psutil.virtual_memory()
            
            return {
                "utilization_percent": cpu_percent,
                "count": cpu_count,
                "memory": {
                    "total_gb": round(mem.total / 1024**3, 2),
                    "used_gb": round(mem.used / 1024**3, 2),
                    "available_gb": round(mem.available / 1024**3, 2),
                    "utilization_percent": round(mem.percent, 2)
                }
            }
        except Exception as e:
            print(f"⚠️  获取 CPU 指标失败: {e}")
            return {}
    
    def check_vllm_health(self) -> Dict:
        """
        检查 vLLM 服务健康状态
        
        Returns:
            Dict: 健康状态信息
        """
        try:
            response = requests.get(f"{self.vllm_url}/health", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "url": self.vllm_url,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                return {
                    "status": "unhealthy",
                    "url": self.vllm_url,
                    "status_code": response.status_code
                }
        except requests.exceptions.RequestException as e:
            return {
                "status": "unreachable",
                "url": self.vllm_url,
                "error": str(e)
            }
    
    def get_all_metrics(self) -> Dict:
        """获取所有指标"""
        uptime = time.time() - self.start_time
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 2),
            "requests": {
                "total": self.total_requests,
                "errors": self.total_errors,
                "success_rate": round((1 - self.total_errors / self.total_requests) * 100, 2) if self.total_requests > 0 else 100.0
            },
            "latency": self.get_latency_stats(),
            "throughput": {
                "requests_per_second_1min": round(self.get_throughput(60), 2),
                "requests_per_second_5min": round(self.get_throughput(300), 2),
                "requests_per_second_15min": round(self.get_throughput(900), 2)
            },
            "gpu": self.get_gpu_metrics(),
            "cpu": self.get_cpu_metrics(),
            "vllm": self.check_vllm_health()
        }


# 全局指标收集器实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(vllm_url: str = "http://localhost:8000") -> MetricsCollector:
    """获取全局指标收集器实例"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(vllm_url=vllm_url)
    return _metrics_collector

