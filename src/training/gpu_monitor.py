#!/usr/bin/env python3
"""
GPU 监控回调类
功能：
1. 实时监控 GPU 使用率、显存使用、温度等指标
2. 将指标记录到 TensorBoard
3. 定期打印 GPU 状态
"""

import torch
import time
from transformers import TrainerCallback
from typing import Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️  警告: pynvml 未安装，将使用基础 GPU 监控（仅显存）")
    print("   安装 pynvml 以获得完整监控: pip install nvidia-ml-py3")


class GPUMonitorCallback(TrainerCallback):
    """GPU 监控回调类"""
    
    def __init__(self, log_interval: int = 10, enable_tensorboard: bool = True):
        """
        初始化 GPU 监控回调
        
        Args:
            log_interval: 打印 GPU 状态的间隔（步数）
            enable_tensorboard: 是否将指标记录到 TensorBoard
        """
        self.log_interval = log_interval
        self.enable_tensorboard = enable_tensorboard
        self.step_count = 0
        self.writer = None
        
        # 使用实例变量跟踪 NVML 是否可用
        self.pynvml_available = PYNVML_AVAILABLE
        self.handles = []
        
        # 初始化 NVML（如果可用）
        if self.pynvml_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
                print(f"✅ GPU 监控已初始化，检测到 {self.device_count} 个 GPU")
            except Exception as e:
                print(f"⚠️  NVML 初始化失败: {e}，将使用基础监控")
                self.pynvml_available = False
                self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                print(f"✅ GPU 监控已初始化（基础模式），检测到 {self.device_count} 个 GPU")
        else:
            self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(f"✅ GPU 监控已初始化（基础模式），检测到 {self.device_count} 个 GPU")
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """训练开始时初始化 TensorBoard writer"""
        if self.enable_tensorboard and hasattr(args, 'logging_dir'):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=args.logging_dir)
                print(f"📊 GPU 监控指标将记录到 TensorBoard: {args.logging_dir}")
            except ImportError:
                print("⚠️  TensorBoard 未安装，GPU 指标将不会记录到 TensorBoard")
                self.enable_tensorboard = False
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """每次日志记录时监控 GPU"""
        if logs is None:
            return
        
        self.step_count = state.global_step
        
        # 获取 GPU 指标
        gpu_metrics = self._get_gpu_metrics()
        
        # 记录到 TensorBoard
        if self.enable_tensorboard and self.writer is not None:
            for gpu_id, metrics in gpu_metrics.items():
                for metric_name, value in metrics.items():
                    if value is not None:
                        self.writer.add_scalar(
                            f'gpu/{gpu_id}/{metric_name}',
                            value,
                            self.step_count
                        )
        
        # 定期打印 GPU 状态
        if self.step_count % self.log_interval == 0:
            self._print_gpu_status(gpu_metrics)
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """训练结束时关闭 TensorBoard writer"""
        if self.writer is not None:
            self.writer.close()
        
        if self.pynvml_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _get_gpu_metrics(self) -> dict:
        """获取所有 GPU 的指标"""
        gpu_metrics = {}
        
        if not torch.cuda.is_available():
            return gpu_metrics
        
        for gpu_id in range(self.device_count):
            metrics = {}
            
            # 基础指标（使用 PyTorch）
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                # 显存使用（MB）
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
                memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2
                
                metrics['memory_allocated_mb'] = memory_allocated
                metrics['memory_reserved_mb'] = memory_reserved
                metrics['memory_total_mb'] = memory_total
                metrics['memory_usage_percent'] = (memory_allocated / memory_total * 100) if memory_total > 0 else 0
                metrics['memory_reserved_percent'] = (memory_reserved / memory_total * 100) if memory_total > 0 else 0
            
            # 高级指标（使用 NVML，如果可用）
            if self.pynvml_available and gpu_id < len(self.handles):
                try:
                    handle = self.handles[gpu_id]
                    
                    # GPU 使用率
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics['utilization_gpu_percent'] = util.gpu
                    metrics['utilization_memory_percent'] = util.memory
                    
                    # 温度
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics['temperature_celsius'] = temp
                    except:
                        metrics['temperature_celsius'] = None
                    
                    # 功耗
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                        metrics['power_usage_watts'] = power
                        metrics['power_limit_watts'] = power_limit
                        metrics['power_usage_percent'] = (power / power_limit * 100) if power_limit > 0 else None
                    except:
                        metrics['power_usage_watts'] = None
                        metrics['power_limit_watts'] = None
                        metrics['power_usage_percent'] = None
                    
                    # 显存信息（NVML 版本，更准确）
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        metrics['memory_used_mb_nvml'] = mem_info.used / 1024**2
                        metrics['memory_free_mb_nvml'] = mem_info.free / 1024**2
                        metrics['memory_total_mb_nvml'] = mem_info.total / 1024**2
                    except:
                        pass
                    
                except Exception as e:
                    # NVML 查询失败，使用基础指标
                    pass
            
            gpu_metrics[f'gpu_{gpu_id}'] = metrics
        
        return gpu_metrics
    
    def _print_gpu_status(self, gpu_metrics: dict):
        """打印 GPU 状态"""
        if not gpu_metrics:
            return
        
        print("\n" + "="*60)
        print(f"📊 GPU 状态 (Step {self.step_count})")
        print("="*60)
        
        for gpu_id, metrics in gpu_metrics.items():
            print(f"\n{gpu_id.upper()}:")
            
            # 显存信息
            if 'memory_allocated_mb' in metrics:
                allocated = metrics['memory_allocated_mb']
                reserved = metrics['memory_reserved_mb']
                total = metrics['memory_total_mb']
                usage_pct = metrics.get('memory_usage_percent', 0)
                
                print(f"  显存: {allocated:.1f}MB / {total:.1f}MB ({usage_pct:.1f}%)")
                print(f"  预留: {reserved:.1f}MB ({metrics.get('memory_reserved_percent', 0):.1f}%)")
            
            # GPU 使用率
            if 'utilization_gpu_percent' in metrics:
                print(f"  GPU 使用率: {metrics['utilization_gpu_percent']}%")
                print(f"  显存使用率: {metrics['utilization_memory_percent']}%")
            
            # 温度
            if 'temperature_celsius' in metrics and metrics['temperature_celsius'] is not None:
                print(f"  温度: {metrics['temperature_celsius']}°C")
            
            # 功耗
            if 'power_usage_watts' in metrics and metrics['power_usage_watts'] is not None:
                power = metrics['power_usage_watts']
                limit = metrics.get('power_limit_watts')
                pct = metrics.get('power_usage_percent')
                if limit:
                    print(f"  功耗: {power:.1f}W / {limit:.1f}W ({pct:.1f}%)" if pct else f"  功耗: {power:.1f}W / {limit:.1f}W")
                else:
                    print(f"  功耗: {power:.1f}W")
        
        print("="*60 + "\n")

