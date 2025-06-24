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


WORKER_LLM_MODEL = "gpt-4o-2024-11-20"
WORKER_REASONING_MODEL = "gpt-4o-2024-11-20"
WORKER_LLM_MODEL = "/mnt/public/algm/models/Qwen3-4B"
WORKER_REASONING_MODEL = "/mnt/public/algm/models/Qwen3-4B"
# WORKER_LLM_MODEL = "/mnt/public/algm/models/Qwen2.5-3B-Instruct"
# WORKER_REASONING_MODEL = "/mnt/public/algm/models/Qwen2.5-3B-Instruct"
# WORKER_LLM_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_wikitablequestions_1_to_600_llava_cot_1_to_200_wthink_3e"
# WORKER_REASONING_MODEL = "/mnt/public/algm/yzy/train_repos/LLaMA-Factory/saves/qwen3-4b/full/sft/hotpotqa_1_to_1200_wikitablequestions_1_to_600_llava_cot_1_to_200_wthink_3e"

worker_model_platform = ModelPlatformType.VLLM
# worker_model_config_dict = {"temperature": 0}
worker_model_config_dict = {"temperature": 0, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
# worker_url = "http://127.0.0.1:8001/v1"
worker_url = "http://59.110.169.144:39929/v1"
# worker_model_platform = ModelPlatformType.OPENAI
# worker_model_config_dict = {"temperature": 0}
# worker_url = None

PIPELINE_LLM_MODEL = "gpt-4o-2024-11-20"
PIPELINE_REASONING_MODEL = "gpt-4o-2024-11-20"
pipeline_model_platform = ModelPlatformType.OPENAI
pipeline_model_config_dict = {"temperature": 0}
# pipeline_model_config_dict = {"temperature": 0, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
# pipeline_url = "http://127.0.0.1:8001/v1"
pipeline_url = None


def construct_agent_list() -> List[Dict[str, Any]]:       
    reasoning_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_REASONING_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url
    )

    email_agent_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_LLM_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url
    )

    office_agent_model = ModelFactory.create(
        model_platform=worker_model_platform,
        model_type=WORKER_LLM_MODEL,
        model_config_dict=worker_model_config_dict,
        url=worker_url
    )

    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    email_toolkit = EmailToolkit()
    office_toolkit = OfficeToolkit()

    email_agent = OwlWorkforceChatAgent(
"""
You are an assistant specialized in analyzing emails and meeting information. You can access Outlook to retrieve email and meeting information.

tips:
- If retrieving meeting schedules, please use the get_meetings_on_specific_day tool
- If the user doesn't clearly specify the time range for emails, please retrieve emails from the last week
- If retrieving meeting schedules, please return detailed meeting schedule information in the results
- If analyzing and processing email content, please first return detailed information for each email: title, sender, recipient, sending time, email content. Finally return the analysis conclusion.
""",
        # """
        # 你是一个专门负责分析邮件和会议信息的助手。你可以通过访问outlook来获取邮件和会议信息。

        # tips:
        # - 如果是获取会议日程，请使用get_meetings_on_specific_day工具
        # - 如果用户没有明确表明获取多少时间范围内的邮件，请获取最近一个星期的邮件
        # - 如果是获取会议日程，请在结果中返回会议日程的详细信息
        # - 如果是分析处理邮件内容，请先在结果中返回各邮件的详细信息：标题，发件人，收件人，发送时间，邮件内容。最后返回分析结论。
        # """,
        model=email_agent_model,
        tools=[
            *email_toolkit.get_tools(),
            FunctionTool(code_runner_toolkit.execute_code),
        ]
    )

    office_agent = OwlWorkforceChatAgent(
"""
You are an assistant specialized in analyzing Office document content. You can:
- Detect all currently open Office documents (Word, Excel, PowerPoint)
- Extract and analyze Office document content

Note:
- Return as complete document content as possible, including absolute file path, title, and document content summary
- Determine which documents may be related to which to-do items, associate them with the to-do items in the work plan, check how much work remains in the documents, and provide a detailed plan for completing the documents; if the document content is not related to the task, there's no need to include it in the work plan
- If the user doesn't provide a clear work plan, please provide a possible work plan based on the document content and the user's likely work nature
- Work plans should not be specific to a certain time point, but rather rough to the level of morning/afternoon
""",
        # """
        # 你是一个专门负责分析Office文档内容的助手。你可以：
        # - 检测当前打开的所有Office文档（Word、Excel、PowerPoint）
        # - 提取和分析Office文档内容

        # 注意：
        # - 返回文档尽可能完整的内容，包含文件绝对路径，标题，文件内容摘要
        # - 判断文档可能与哪个代办事项相关，在工作计划中与该代办事项相关联，查看文档还有多少工作量，并给出完成文档的详细计划；如果文档内容与任务不相关，无需纳入工作计划
        # - 如果用户没有提供明确的工作计划，请跟根据文档内容和用户可能的工作性质，给出可能的工作计划
        # - 工作计划不要具体到某个时间点，而是粗略到上午下午这种粒度
        # """,
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
            model_platform=pipeline_model_platform,
            model_type=PIPELINE_REASONING_MODEL,
            model_config_dict=pipeline_model_config_dict,
            url=pipeline_url
        )
    }
    
    task_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=pipeline_model_platform,
            model_type=PIPELINE_LLM_MODEL,
            model_config_dict=pipeline_model_config_dict,
            url=pipeline_url
        )
    }
    
    answerer_agent_kwargs = {
        "model": ModelFactory.create(
            model_platform=pipeline_model_platform,
            model_type=PIPELINE_LLM_MODEL,
            model_config_dict=pipeline_model_config_dict,
            url=pipeline_url
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

Please complete the task in <task>, you need to:
- If <guideline> has content, decompose <task> according to the instructions in <guideline>
- Complete <task> based on the information provided in <task> and <tips>
"""

    date_prompt = "Today is 2025-06-08"
    guideline_prompt = """- Check my meeting schedule for tomorrow
- Check my emails from the last week (up to today)
- Determine tomorrow's to-do items based on tomorrow's meeting schedule and email content
- Determine the document content related to tomorrow's to-do items based on the Office documents already opened on my computer
- Based on all relevant information, provide a possible work plan for tomorrow
""" if with_guideline else ""
    tips_prompt = """- Complete <task> with as few steps as possible
- When creating a work plan, consider both meeting schedule, email content, and document content opened on my computer
- If the user doesn't clearly specify the time range for emails, please retrieve emails from the last week
- When creating a work plan, refer to user profile: users like to prepare materials for meetings in the morning and learn new knowledge in the afternoon
- When creating a work plan, if there's a need to check related documents, please provide the absolute path of the related documents and explain why these documents are needed
- Tomorrow's meeting schedule information must be included in the work plan (specific to time points); other work plans should not be specific to a certain time point, but rather rough to the level of morning/afternoon
- Final work plan should be output in markdown format
- Answer in Chinese
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
        # "planning tomorrow's work",
        with_guideline=False
        # with_guideline=True
    )