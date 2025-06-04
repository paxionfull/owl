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

import os
import json
from typing import List, Dict, Any
from loguru import logger
from utils import OwlWorkforceChatAgent, OwlGaiaWorkforce
from utils.gaia import GAIABenchmark
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from examples.overwrite_modules.browser_user_toolkit import BrowserUseToolkit

LLM_MODEL = "gpt-4o-2024-11-20"
REASONING_MODEL = "gpt-4o-2024-11-20"


def construct_agent_list() -> List[Dict[str, Any]]:

    web_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    document_processing_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    reasoning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=REASONING_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    image_analysis_model = ModelFactory.create( 
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    audio_reasoning_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=REASONING_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=LLM_MODEL,
        model_config_dict={"temperature": 0},
    )
    
    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=REASONING_MODEL,
        model_config_dict={"temperature": 0},
    )
    

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp")
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)
    video_analysis_toolkit = VideoAnalysisToolkit(download_directory="tmp/video")
    # audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=audio_reasoning_model)
    audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=None)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    browser_simulator_toolkit = AsyncBrowserToolkit(headless=True, cache_dir=f"tmp/browser", planning_agent_model=planning_agent_model, web_agent_model=web_agent_model)
    excel_toolkit = ExcelToolkit()
    browser_user_toolkit = BrowserUseToolkit(headless=True)


    web_agent = OwlWorkforceChatAgent(
"""
You are a helpful assistant that can search the web, extract webpage content, simulate browser actions, and provide relevant information to solve the given task.
Keep in mind that:
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- You can also simulate browser actions to get more information or verify the information you have found.
- If extracting webpage content cannot provide the detailed information about the answer, you should use browser simulation to get more information, else you don't need to use browser simulation.
- Browser simulation is also helpful for finding target URLs. Browser simulation operations do not necessarily need to find specific answers, but can also help find web page URLs that contain answers (usually difficult to find through simple web searches). You can find the answer to the question by performing subsequent operations on the URL, such as extracting the content of the webpage.
- Do not solely rely on document tools or browser simulation to find the answer, you should combine document tools and browser simulation to comprehensively process web page information. Some content may need to do browser simulation to get, or some content is rendered by javascript.
- In your response, you should mention the urls you have visited and processed.

Here are some tips that help you perform web search:
- Never add too many keywords in your search query! Some detailed results need to perform browser interaction to get, not using search toolkit.
- If the question is complex, search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding official sources rather than direct answers.
  For example, as for the question "What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website?", your first search term must be coarse-grained like "National Geographic YouTube" to find the youtube website first, and then try other fine-grained search terms step-by-step to find more urls.
- The results you return do not have to directly answer the original question, you only need to collect relevant information.
- If there are multiple documents to be processed, you should process all the documents in the list at once, do not process one by one.
""",
# - If extracting webpage content cannot provide the detailed information about the answer, you should use browser simulation to get more information, else you don't need to use browser simulation.
        model=web_model,
        tools=[
            # FunctionTool(search_toolkit.search_google),
            FunctionTool(search_toolkit.search_serper_api),
            FunctionTool(search_toolkit.search_wiki),
            FunctionTool(search_toolkit.search_wiki_revisions),
            FunctionTool(search_toolkit.search_archived_webpage),
            FunctionTool(document_processing_toolkit.extract_document_content),
            # FunctionTool(browser_simulator_toolkit.browse_url),
            FunctionTool(browser_user_toolkit.browse_url),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
        ]
    )
    
    document_processing_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video.",
        document_processing_model,
        tools=[
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(image_analysis_toolkit.ask_question_about_image),
            FunctionTool(audio_analysis_toolkit.ask_question_about_audio),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
            FunctionTool(code_runner_toolkit.execute_code),
        ]
    )
    
    reasoning_coding_agent = OwlWorkforceChatAgent(
        "You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them.",
        reasoning_model,
        tools=[
            FunctionTool(code_runner_toolkit.execute_code),
            FunctionTool(excel_toolkit.extract_excel_content),
            FunctionTool(document_processing_toolkit.extract_document_content),
        ]
    )

    agent_list = []
    
    web_agent_dict = {
        "name": "Web Agent",
        "description": "A helpful assistant that can search the web, extract webpage content, simulate browser actions, and retrieve relevant information.",
        "agent": web_agent
    }
    
    document_processing_agent_dict = {
        "name": "Document Processing Agent",
        "description": "A helpful assistant that can process a variety of local and remote documents, including pdf, docx, images, audio, and video, etc.",
        "agent": document_processing_agent
    }
    
    reasoning_coding_agent_dict = {
        "name": "Reasoning Coding Agent",
        "description": "A helpful assistant that specializes in reasoning, coding, and processing excel files. However, it cannot access the internet to search for information. If the task requires python execution, it should be informed to execute the code after writing it.",
        "agent": reasoning_coding_agent
    }

    agent_list.append(web_agent_dict)
    agent_list.append(document_processing_agent_dict)
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
        "Gaia Workforce",
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


def process_single_task_index(
    task_idx: int,
    level: int,
    on: str,
    save_result: bool,
    max_tries: int,
    max_replanning_tries: int,
    data_dir: str,
    result_path: str,
    thread_id: int
) -> Dict[str, Any]:
    """处理单个任务索引，用于并行执行"""
    
    # 为每个线程创建独立的workforce和benchmark
    workforce = construct_workforce()
    
    benchmark = GAIABenchmark(
        data_dir=data_dir,
        save_to=f"{result_path}_thread_{thread_id}.json",
    )
    
    logger.info(f"Thread {thread_id}: Processing task index {task_idx}")
    
    try:
        result = benchmark.run_workforce_with_retry(
            workforce,
            on=on,
            level=level,
            idx=[task_idx],  # 只处理单个任务
            save_result=save_result,
            max_tries=max_tries,
            max_replanning_tries=max_replanning_tries,
        )
            
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'result': result,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Thread {thread_id}: Error processing task {task_idx}: {e}")
        return {
            'task_idx': task_idx,
            'thread_id': thread_id,
            'error': str(e),
            'success': False
        }


def evaluate_on_gaia():
    
    LEVEL = 1
    on="valid"
    SAVE_RESULT = True
    # MAX_TRIES = 3
    MAX_TRIES = 1
    PARALLEL = True  # 新增：是否启用并行处理
    MAX_WORKERS = 20  # 新增：最大并行线程数
    
    SAVE_RESULT_PATH = f"results/workforce/workforce_{LEVEL}_pass{MAX_TRIES}_gpt4o.json"
    test_idx = list(range(53))
    # test_idx = [30]
    # test_idx = [3]
    # test_idx = [13]
    # test_idx = [16]
    # test_idx = [15]  # hard
    # test_idx = [1]

    if os.path.exists(f"tmp/"):
        shutil.rmtree(f"tmp/")
    
    if PARALLEL and len(test_idx) > 1:
        # 并行处理模式
        logger.info(f"Using parallel processing with {MAX_WORKERS} workers for {len(test_idx)} tasks")
        
        # 确保结果目录存在
        os.makedirs(os.path.dirname(SAVE_RESULT_PATH), exist_ok=True)
        
        all_results = []
        total_correct = 0
        total_tasks = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task_idx in enumerate(test_idx):
                future = executor.submit(
                    process_single_task_index,
                    task_idx=task_idx,
                    level=LEVEL,
                    on=on,
                    save_result=SAVE_RESULT,
                    max_tries=MAX_TRIES,
                    max_replanning_tries=2,
                    data_dir="data/gaia",
                    result_path=SAVE_RESULT_PATH.replace('.json', ''),
                    thread_id=i
                )
                future_to_task[future] = task_idx
            
            # 收集结果
            for future in as_completed(future_to_task):
                task_idx = future_to_task[future]
                try:
                    task_result = future.result()
                    if task_result['success']:
                        result = task_result['result']
                        all_results.extend(result.get('results', []))
                        total_correct += result.get('correct', 0)
                        total_tasks += result.get('total', 0)
                        logger.success(f"Task {task_idx} completed. Correct: {result.get('correct', 0)}/{result.get('total', 0)}")
                    else:
                        logger.error(f"Task {task_idx} failed: {task_result.get('error', 'Unknown error')}")
                        total_tasks += 1  # 仍然计入总数
                except Exception as exc:
                    logger.error(f"Task {task_idx} generated an exception: {exc}")
                    total_tasks += 1
        
        # 保存合并后的结果
        if SAVE_RESULT:
            final_result = {
                "total": total_tasks,
                "correct": total_correct,
                "accuracy": total_correct / total_tasks if total_tasks > 0 else 0,
                "results": all_results
            }
            
            with open(SAVE_RESULT_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            # 清理线程特定的结果文件
            for i in range(len(test_idx)):
                thread_file = f"{SAVE_RESULT_PATH.replace('.json', '')}_thread_{i}.json"
                if os.path.exists(thread_file):
                    os.remove(thread_file)
        
        logger.success(f"Parallel processing completed. Correct: {total_correct}, Total: {total_tasks}")
        logger.success(f"Accuracy: {total_correct / total_tasks if total_tasks > 0 else 0}")
        
    else:
        # 原始顺序处理模式
        logger.info("Using sequential processing")
        
        benchmark = GAIABenchmark(
            data_dir="data/gaia",
            save_to=SAVE_RESULT_PATH,
        )
        
        workforce = construct_workforce()

        result = benchmark.run_workforce_with_retry(
            workforce,
            on=on,
            level=LEVEL,
            idx=test_idx,
            save_result=SAVE_RESULT,
            max_tries=MAX_TRIES,
            max_replanning_tries=2,
        )
        
        logger.success(f"Correct: {result['correct']}, Total: {result['total']}")
        logger.success(f"Accuracy: {result['accuracy']}")


if __name__ == "__main__":
    evaluate_on_gaia()

