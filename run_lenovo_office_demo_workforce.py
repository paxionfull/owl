# -*- coding: utf-8 -*-
import sys
import os
import locale

# 修复Windows控制台编码问题
if sys.platform == "win32":
    # 设置控制台编码为UTF-8
    os.system("chcp 65001 > nul")
    # 重新配置stdout和stderr
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    # 设置locale
    try:
        locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'Chinese_China.65001')
        except:
            pass

from camel.toolkits import (
    VideoAnalysisToolkit,
    SearchToolkit,
    CodeExecutionToolkit,
    ImageAnalysisToolkit,
    DocumentProcessingToolkit,
    AudioAnalysisToolkit,
    AsyncBrowserToolkit,
    ExcelToolkit,
    FunctionTool
)
from camel.models import ModelFactory
from camel.types import(
    ModelPlatformType,
    ModelType
)
from camel.tasks import Task
from dotenv import load_dotenv

load_dotenv(override=True)

import json
from typing import List, Dict, Any
from loguru import logger
from utils import OwlWorkforceChatAgent, OwlGaiaWorkforce
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from examples.overwrite_modules.email_toolkit import EmailToolkit
from examples.overwrite_modules.office_toolkit import OfficeToolkit


LLM_MODEL = "gpt-4o-2024-11-20"
REASONING_MODEL = "gpt-4o-2024-11-20"


def construct_agent_list() -> List[Dict[str, Any]]:        
    reasoning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=REASONING_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    email_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )

    office_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    

    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    email_toolkit = EmailToolkit()
    office_toolkit = OfficeToolkit()

    email_agent = OwlWorkforceChatAgent(
"""
你是一个专门负责分析邮件和会议信息的助手。你可以通过访问outlook来获取邮件和会议信息。

tips:
- 如果是获取会议日程，请使用get_meetings_on_specific_day工具
- 如果用户没有明确表明获取多少时间范围内的邮件，请获取最近一个星期的邮件
- 如果是获取会议日程，请在结果中返回会议日程的详细信息
- 如果是分析处理邮件内容，请先在结果中返回各邮件的详细信息：标题，发件人，收件人，发送时间，邮件内容。最后返回分析结论。
""",
        model=email_agent_model,
        tools=[
            *email_toolkit.get_tools(),
            FunctionTool(code_runner_toolkit.execute_code),
        ]
    ) 

# - 理解文档中体现的工作重点和优先级
# - 识别文档中可能的截止时间、里程碑和依赖关系

# 请确保：
# - 首先获取所有正在运行的Office文档路径
# - 详细解析每个文档的内容，理解工作上下文
# - 识别文档中的任务列表、项目进度、待办事项
# - 分析文档反映的当前工作重点和下一步计划

    office_agent = OwlWorkforceChatAgent(
"""
你是一个专门负责分析Office文档内容的助手。你可以：
- 检测当前打开的所有Office文档（Word、Excel、PowerPoint）
- 提取和分析Office文档内容

注意：
- 返回文档尽可能完整的内容，包含文件绝对路径，标题，文件内容摘要
- 判断文档可能与哪个代办事项相关，在工作计划中与该代办事项相关联，查看文档还有多少工作量，并给出完成文档的详细计划；如果文档内容与任务不相关，无需纳入工作计划
- 如果用户没有提供明确的工作计划，请跟根据文档内容和用户可能的工作性质，给出可能的工作计划
- 工作计划不要具体到某个时间点，而是粗略到上午下午这种粒度
""",
        model=office_agent_model,
        tools=[
            *office_toolkit.get_tools(),
        ]
    ) 

    reasoning_coding_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them.",
        reasoning_model,
        tools=[
            FunctionTool(code_runner_toolkit.execute_code),
            # FunctionTool(document_processing_toolkit.extract_document_content),
        ]
    )

    agent_list = []

    
    email_agent_dict = {
        "name": "Email Agent",
        "description": "A helpful assistant that can analyze emails and meetings and extract task information",
        "agent": email_agent
    }
    
    office_agent_dict = {
        "name": "Office Agent",
        "description": "A helpful assistant that can analyze office documents and extract relevant information about the task",
        "agent": office_agent
    }
    
    reasoning_coding_agent_dict = {
        "name": "Reasoning Coding Agent",
        "description": "A helpful assistant that specializes in reasoning, coding, and processing excel files. However, it cannot access the internet to search for information. If the task requires python execution, it should be informed to execute the code after writing it.",
        "agent": reasoning_coding_agent
    }

    # agent_list.append(web_agent_dict)
    agent_list.append(email_agent_dict)
    agent_list.append(office_agent_dict)
    agent_list.append(reasoning_coding_agent_dict)
    return agent_list


def construct_workforce() -> OwlGaiaWorkforce:
    
    coordinator_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=REASONING_MODEL,
            model_config_dict={"temperature": 0},
        )
    }
    
    task_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        )
    }
    
    answerer_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=LLM_MODEL,
            model_config_dict={"temperature": 0},
        )
    }
    
    workforce = OwlGaiaWorkforce(
        "Lenovo Office Demo Workforce",
        task_agent_kwargs=task_agent_kwargs,
        coordinator_agent_kwargs=coordinator_agent_kwargs,
        answerer_agent_kwargs=answerer_agent_kwargs
    )

    agent_list = construct_agent_list()
    
    for agent_dict in agent_list:
        workforce.add_single_agent_worker(
            agent_dict["description"],
            worker=agent_dict["agent"],
        )

    return workforce


def process_single_prompt(
    prompt: str,
    file_paths: List[str] = None,
    prompt_id: str = None,
    max_tries: int = 1,
    max_replanning_tries: int = 2,
    thread_id: int = 0
) -> Dict[str, Any]:
    """处理单个自定义prompt"""
    
    # 为每个线程创建独立的workforce
    workforce = construct_workforce()
    
    logger.info(f"Thread {thread_id}: Processing prompt: {prompt[:100]}...")
    
    try:
        # 创建任务
        task = Task(
            content=prompt,
            id=prompt_id or f"custom_task_{thread_id}",
            additional_info=json.dumps({"file_paths": file_paths or []})  # 转换为JSON字符串
        )
        
        # 执行任务
        response = workforce.process_task(
            task=task,
            max_replanning_tries=max_replanning_tries
        )
        
        # 将响应转换为可序列化的格式
        serializable_response = {
            "task_id": response.id,
            "content": response.content,
            "result": response.result,
            "state": str(response.state),
            "additional_info": response.additional_info
        }
            
        return {
            'prompt_id': prompt_id,
            'thread_id': thread_id,
            'prompt': prompt,
            'response': serializable_response,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing prompt: {e}")
        return {
            'prompt_id': prompt_id,
            'thread_id': thread_id,
            'prompt': prompt,
            'error': str(e),
            'success': False
        }


def run_demo_prompts():
    """运行联想办公室演示的自定义prompts"""
    
    SAVE_RESULT = True
    MAX_TRIES = 1
    SAVE_RESULT_PATH = "results/lenovo_office_demo/demo_results.json"
    
    # 定义测试prompt
    test_prompt = {
        "id": "market_research",
        "prompt": "请帮我研究一下2024年全球笔记本电脑市场的最新趋势，包括主要品牌的市场份额、技术发展方向以及消费者偏好变化。",
        "files": []
    }

    # 清理临时目录
    if os.path.exists(f"tmp/"):
        shutil.rmtree(f"tmp/")
    
    logger.info(f"Processing prompt: {test_prompt['id']}")
    
    # 处理单个prompt
    result = process_single_prompt(
        prompt=test_prompt["prompt"],
        file_paths=test_prompt.get("files", []),
        prompt_id=test_prompt["id"],
        max_tries=MAX_TRIES,
        max_replanning_tries=2,
        thread_id=0
    )
    
    # 显示结果
    if result['success']:
        logger.success(f"Prompt '{test_prompt['id']}' completed successfully")
        if result.get('response'):
            logger.info(f"Response preview: {str(result['response'])[:200]}...")
    else:
        logger.error(f"Prompt '{test_prompt['id']}' failed: {result.get('error', 'Unknown error')}")
    
    # 保存结果
    if SAVE_RESULT:
        final_result = {
            "total_prompts": 1,
            "successful_prompts": 1 if result['success'] else 0,
            "results": [result]
        }
        
        os.makedirs(os.path.dirname(SAVE_RESULT_PATH), exist_ok=True)
        with open(SAVE_RESULT_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Processing completed. Success: {result['success']}")


def run_custom_prompt(prompt: str, file_paths: List[str] = None, with_guideline: bool = True):
    """运行单个自定义prompt的便捷函数"""
    
    logger.info(f"Processing custom prompt: {prompt[:100]}...")


    prompt_template = """
<task>
{task_prompt}
</task>
<date>
{date_prompt}
</date>
<guideline>
{guideline_prompt}
</guideline>
<tips>
{tips_prompt}
</tips>

请完成<task>中的任务，你需要：
- 如果<guideline>有内容，按照<guideline>中的方式拆解<task>
- 按照<task>和<tips>中提供的信息完成<task>
"""

    date_prompt = "今天是2025-06-08"
    guideline_prompt = """- 查看我的邮件会议日程看看明天有什么会议
- 查看我最近一个星期(截至今天)的邮件
- 根据明天会议日程和邮件内容确定明天代办事项
- 根据我电脑上已经打开的办公文档确定与明天代办相关的文档内容
- 综合所有相关信息，给出明天的工作计划
""" if with_guideline else ""
    tips_prompt = """- 用尽可能少的步骤完成<task>
- 制定工作计划时，要综合查看会议日程，邮件内容和电脑上打开的文档内容
- 如果用户没有明确表明获取多少时间范围内的邮件，请获取最近一个星期的邮件
- 制定工作计划时参考用户画像：用户喜欢上午准备开会相关的资料，下午学习新知识
- 制定工作计划时，如果存在需要查看相关文档，请一定要给出相关文档的绝对路径，并说明为什么需要这些文档
- 明天的会议日程信息必须包含在工作计划中（具体到时间点）；其余工作计划不要具体到某个时间点，而是粗略到上午下午这种粒度
- 最终工作计划使用markdown格式输出
- 回答使用中文
"""
    prompt = prompt_template.format(
        date_prompt=date_prompt,
        task_prompt=prompt,
        guideline_prompt=guideline_prompt,
        tips_prompt=tips_prompt
    )
    
    if os.path.exists(f"tmp/"):
        shutil.rmtree(f"tmp/")
    
    result = process_single_prompt(
        prompt=prompt,
        file_paths=file_paths or [],
        prompt_id="custom_single",
        max_tries=1,
        max_replanning_tries=2,
        thread_id=0
    )
    
    if result['success']:
        logger.success("Custom prompt completed successfully")
        logger.info(f"Response: {result['response']}")
        return result['response']
    else:
        logger.error(f"Custom prompt failed: {result.get('error', 'Unknown error')}")
        return None


if __name__ == "__main__":
    # 配置日志输出到文件
    import datetime
    log_filename = f"run_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置loguru同时输出到控制台和文件
    logger.add(
        log_filename,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8"
    )
    
    print(f"日志将同时输出到控制台和文件: {log_filename}")
    
    # 可以选择运行演示prompts或单个自定义prompt
    
    # 选项1: 运行预定义的演示prompts
    # run_demo_prompts()
    
    # 选项2: 运行单个自定义prompt (取消注释以使用)
    custom_response = run_custom_prompt(
        # "查看我电脑上打开的办公文档，并总结我的工作内容, 进而给出第二天详细的工作计划, 用中文回答"
        # "今天是2025-06-08，查看我的邮件会议日程看看明天有什么会议， 同时查看我最近一个月(截至今天)的邮件，根据我电脑上已经打开的办公文档帮我确定明天的工作计划. 用尽可能少的步骤实现，用中文回答"
        # "今天是2025-06-08\n为我规划一下明日的工作计划\n用中文回答"
        "为我规划一下明日的工作计划",
        with_guideline=False
        # with_guideline=True
    )