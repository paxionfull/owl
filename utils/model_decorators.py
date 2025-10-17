"""
模型装饰器
用于拦截模型API调用并记录token使用情况
"""
import functools
from typing import Any, Dict, Optional
from loguru import logger
from utils.token_tracker import global_token_tracker


def track_token_usage(tracker=None, agent_name: str = "unknown", task_id: Optional[str] = None):
    """
    装饰器：拦截模型API调用并记录token使用情况
    
    Args:
        tracker: Token统计器实例，默认为全局统计器
        agent_name: Agent名称
        task_id: 任务ID
    """
    if tracker is None:
        tracker = global_token_tracker
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, messages, *args, **kwargs):
            # 调用原始方法
            response = func(self, messages, *args, **kwargs)
            
            # 提取token使用信息
            try:
                if hasattr(response, 'usage') and response.usage:
                    usage_dict = {}
                    
                    # 处理不同的usage格式
                    if hasattr(response.usage, '__dict__'):
                        usage_dict = response.usage.__dict__
                    elif hasattr(response.usage, 'prompt_tokens'):
                        usage_dict = {
                            'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                            'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                            'total_tokens': getattr(response.usage, 'total_tokens', 0)
                        }
                    elif hasattr(response.usage, 'input_tokens'):
                        usage_dict = {
                            'input_tokens': getattr(response.usage, 'input_tokens', 0),
                            'output_tokens': getattr(response.usage, 'output_tokens', 0),
                            'total_tokens': getattr(response.usage, 'total_tokens', 0)
                        }
                    
                    # 获取模型名称
                    model_name = getattr(self, 'model_type', 'unknown')
                    if hasattr(model_name, 'value'):
                        model_name = model_name.value
                    
                    # 获取Agent名称
                    current_agent_name = agent_name
                    if hasattr(self, 'role_name'):
                        current_agent_name = getattr(self, 'role_name', agent_name)
                    elif hasattr(self, 'name'):
                        current_agent_name = getattr(self, 'name', agent_name)
                    
                    # 记录token使用情况
                    tracker.record_usage(
                        model_name=model_name,
                        agent_name=current_agent_name,
                        usage_dict=usage_dict,
                        task_id=task_id,
                        additional_info={
                            'function_name': func.__name__,
                            'messages_count': len(messages) if messages else 0
                        }
                    )
                    
                    logger.debug(f"Token统计: {model_name} - {current_agent_name} - {usage_dict}")
                    
            except Exception as e:
                logger.warning(f"记录token使用情况时出错: {e}")
            
            return response
        
        return wrapper
    return decorator


def track_async_token_usage(tracker=None, agent_name: str = "unknown", task_id: Optional[str] = None):
    """
    异步装饰器：拦截异步模型API调用并记录token使用情况
    """
    if tracker is None:
        tracker = global_token_tracker
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, messages, *args, **kwargs):
            # 调用原始异步方法
            response = await func(self, messages, *args, **kwargs)
            
            # 提取token使用信息
            try:
                if hasattr(response, 'usage') and response.usage:
                    usage_dict = {}
                    
                    # 处理不同的usage格式
                    if hasattr(response.usage, '__dict__'):
                        usage_dict = response.usage.__dict__
                    elif hasattr(response.usage, 'prompt_tokens'):
                        usage_dict = {
                            'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                            'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                            'total_tokens': getattr(response.usage, 'total_tokens', 0)
                        }
                    elif hasattr(response.usage, 'input_tokens'):
                        usage_dict = {
                            'input_tokens': getattr(response.usage, 'input_tokens', 0),
                            'output_tokens': getattr(response.usage, 'output_tokens', 0),
                            'total_tokens': getattr(response.usage, 'total_tokens', 0)
                        }
                    
                    # 获取模型名称
                    model_name = getattr(self, 'model_type', 'unknown')
                    if hasattr(model_name, 'value'):
                        model_name = model_name.value
                    
                    # 获取Agent名称
                    current_agent_name = agent_name
                    if hasattr(self, 'role_name'):
                        current_agent_name = getattr(self, 'role_name', agent_name)
                    elif hasattr(self, 'name'):
                        current_agent_name = getattr(self, 'name', agent_name)
                    
                    # 记录token使用情况
                    tracker.record_usage(
                        model_name=model_name,
                        agent_name=current_agent_name,
                        usage_dict=usage_dict,
                        task_id=task_id,
                        additional_info={
                            'function_name': func.__name__,
                            'messages_count': len(messages) if messages else 0,
                            'async': True
                        }
                    )
                    
                    logger.debug(f"Token统计(异步): {model_name} - {current_agent_name} - {usage_dict}")
                    
            except Exception as e:
                logger.warning(f"记录异步token使用情况时出错: {e}")
            
            return response
        
        return wrapper
    return decorator


def apply_token_tracking_to_model(model_class, agent_name: str = "unknown", task_id: Optional[str] = None):
    """
    为模型类应用token统计装饰器
    
    Args:
        model_class: 模型类
        agent_name: Agent名称
        task_id: 任务ID
    """
    # 装饰同步方法
    if hasattr(model_class, '_run'):
        model_class._run = track_token_usage(agent_name=agent_name, task_id=task_id)(model_class._run)
    
    # 装饰异步方法
    if hasattr(model_class, '_arun'):
        model_class._arun = track_async_token_usage(agent_name=agent_name, task_id=task_id)(model_class._arun)
    
    logger.info(f"已为模型类 {model_class.__name__} 应用token统计装饰器")


def create_tracked_model_factory(original_factory):
    """
    创建带token统计的模型工厂函数
    
    Args:
        original_factory: 原始模型工厂函数
    """
    def tracked_factory(*args, **kwargs):
        # 调用原始工厂函数
        model = original_factory(*args, **kwargs)
        
        # 为模型应用token统计
        agent_name = kwargs.get('agent_name', 'unknown')
        task_id = kwargs.get('task_id', None)
        
        # 装饰模型的_run和_arun方法
        if hasattr(model, '_run'):
            model._run = track_token_usage(agent_name=agent_name, task_id=task_id)(model._run)
        
        if hasattr(model, '_arun'):
            model._arun = track_async_token_usage(agent_name=agent_name, task_id=task_id)(model._arun)
        
        return model
    
    return tracked_factory


