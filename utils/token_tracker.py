"""
Token使用统计器
用于统计整个任务执行过程中的token使用情况
"""
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger


class TokenUsageTracker:
    """Token使用统计器"""
    
    def __init__(self):
        self.reset()
        self.current_task_id = None
        self.task_start_time = None
    
    def reset(self):
        """重置统计器"""
        self.stats = {
            'total': {
                'input_tokens': 0,
                'output_tokens': 0, 
                'total_tokens': 0,
                'api_calls': 0
            },
            'by_model': {},
            'by_agent': {},
            'by_task': {},
            'detailed_log': []
        }
        logger.info("Token统计器已重置")
    
    def start_task(self, task_id: str):
        """开始任务统计"""
        self.current_task_id = task_id
        self.task_start_time = time.time()
        if task_id not in self.stats['by_task']:
            self.stats['by_task'][task_id] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'api_calls': 0,
                'start_time': self.task_start_time
            }
        logger.info(f"开始统计任务 {task_id} 的token使用情况")
    
    def end_task(self, task_id: str):
        """结束任务统计"""
        if task_id in self.stats['by_task']:
            self.stats['by_task'][task_id]['end_time'] = time.time()
            duration = self.stats['by_task'][task_id]['end_time'] - self.stats['by_task'][task_id]['start_time']
            self.stats['by_task'][task_id]['duration'] = duration
            logger.info(f"任务 {task_id} 完成，耗时 {duration:.2f} 秒")
        self.current_task_id = None
        self.task_start_time = None
    
    def record_usage(self, model_name: str, agent_name: str, usage_dict: Dict[str, Any], 
                    task_id: Optional[str] = None, additional_info: Optional[Dict] = None):
        """记录token使用情况"""
        if not usage_dict:
            return
            
        # 提取token信息
        input_tokens = usage_dict.get('prompt_tokens', usage_dict.get('input_tokens', 0))
        output_tokens = usage_dict.get('completion_tokens', usage_dict.get('output_tokens', 0))
        total_tokens = usage_dict.get('total_tokens', input_tokens + output_tokens)
        
        # 更新总统计
        self.stats['total']['input_tokens'] += input_tokens
        self.stats['total']['output_tokens'] += output_tokens
        self.stats['total']['total_tokens'] += total_tokens
        self.stats['total']['api_calls'] += 1
        
        # 按模型统计
        if model_name not in self.stats['by_model']:
            self.stats['by_model'][model_name] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'api_calls': 0
            }
        self.stats['by_model'][model_name]['input_tokens'] += input_tokens
        self.stats['by_model'][model_name]['output_tokens'] += output_tokens
        self.stats['by_model'][model_name]['total_tokens'] += total_tokens
        self.stats['by_model'][model_name]['api_calls'] += 1
        
        # 按agent统计
        if agent_name not in self.stats['by_agent']:
            self.stats['by_agent'][agent_name] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'api_calls': 0
            }
        self.stats['by_agent'][agent_name]['input_tokens'] += input_tokens
        self.stats['by_agent'][agent_name]['output_tokens'] += output_tokens
        self.stats['by_agent'][agent_name]['total_tokens'] += total_tokens
        self.stats['by_agent'][agent_name]['api_calls'] += 1
        
        # 按任务统计
        current_task = task_id or self.current_task_id
        if current_task:
            if current_task not in self.stats['by_task']:
                self.stats['by_task'][current_task] = {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'api_calls': 0,
                    'start_time': time.time()
                }
            self.stats['by_task'][current_task]['input_tokens'] += input_tokens
            self.stats['by_task'][current_task]['output_tokens'] += output_tokens
            self.stats['by_task'][current_task]['total_tokens'] += total_tokens
            self.stats['by_task'][current_task]['api_calls'] += 1
        
        # 详细日志
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'agent_name': agent_name,
            'task_id': current_task,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'usage_dict': usage_dict,
            'additional_info': additional_info or {}
        }
        self.stats['detailed_log'].append(log_entry)
        
        logger.debug(f"记录token使用: {model_name} - {agent_name} - 输入:{input_tokens}, 输出:{output_tokens}, 总计:{total_tokens}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'summary': self.stats['total'],
            'by_model': self.stats['by_model'],
            'by_agent': self.stats['by_agent'],
            'by_task': self.stats['by_task']
        }
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*60)
        print("                    TOKEN使用统计报告")
        print("="*60)
        
        # 总体统计
        total = self.stats['total']
        print(f"\n📊 总体统计:")
        print(f"   总输入tokens: {total['input_tokens']:,}")
        print(f"   总输出tokens: {total['output_tokens']:,}")
        print(f"   总tokens: {total['total_tokens']:,}")
        print(f"   API调用次数: {total['api_calls']}")
        
        # 按模型统计
        if self.stats['by_model']:
            print(f"\n🤖 按模型统计:")
            for model, stats in self.stats['by_model'].items():
                print(f"   {model}:")
                print(f"     输入: {stats['input_tokens']:,} tokens")
                print(f"     输出: {stats['output_tokens']:,} tokens")
                print(f"     总计: {stats['total_tokens']:,} tokens")
                print(f"     调用: {stats['api_calls']} 次")
        
        # 按Agent统计
        if self.stats['by_agent']:
            print(f"\n👥 按Agent统计:")
            for agent, stats in self.stats['by_agent'].items():
                print(f"   {agent}:")
                print(f"     输入: {stats['input_tokens']:,} tokens")
                print(f"     输出: {stats['output_tokens']:,} tokens")
                print(f"     总计: {stats['total_tokens']:,} tokens")
                print(f"     调用: {stats['api_calls']} 次")
        
        # 按任务统计
        if self.stats['by_task']:
            print(f"\n📋 按任务统计:")
            for task, stats in self.stats['by_task'].items():
                duration = stats.get('duration', 0)
                print(f"   任务 {task}:")
                print(f"     输入: {stats['input_tokens']:,} tokens")
                print(f"     输出: {stats['output_tokens']:,} tokens")
                print(f"     总计: {stats['total_tokens']:,} tokens")
                print(f"     调用: {stats['api_calls']} 次")
                print(f"     耗时: {duration:.2f} 秒")
        
        print("="*60)
    
    def save_detailed_log(self, filepath: str):
        """保存详细日志到文件"""
        log_data = {
            'summary': self.get_summary(),
            'detailed_log': self.stats['detailed_log'],
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细token使用日志已保存到: {filepath}")
    
    def get_cost_estimate(self, pricing: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """估算成本（需要提供定价信息）"""
        if not pricing:
            return {"message": "需要提供模型定价信息才能估算成本"}
        
        cost_estimate = {
            'total_cost': 0,
            'by_model': {},
            'currency': 'USD'
        }
        
        for model, stats in self.stats['by_model'].items():
            if model in pricing:
                input_cost = stats['input_tokens'] * pricing[model].get('input', 0) / 1000
                output_cost = stats['output_tokens'] * pricing[model].get('output', 0) / 1000
                total_cost = input_cost + output_cost
                
                cost_estimate['by_model'][model] = {
                    'input_cost': input_cost,
                    'output_cost': output_cost,
                    'total_cost': total_cost
                }
                cost_estimate['total_cost'] += total_cost
        
        return cost_estimate


# 全局token统计器实例
global_token_tracker = TokenUsageTracker()


