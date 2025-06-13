import sys
sys.path.append("../")

import json
import os
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Tuple, Callable

from tqdm import tqdm
from camel.benchmarks import BaseBenchmark
from camel.models import BaseModelBackend
from camel.tasks import Task
from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from loguru import logger
from .common import extract_pattern, extract_dict_from_str
from .enhanced_role_playing import OwlGaiaRolePlaying, run_society
from .enhanced_workforce import OwlGaiaWorkforce


class MINTBenchmark(BaseBenchmark):
    r"""MINT Benchmark for evaluating AI agents on diverse tasks.

    Args:
        data_dir (str): The directory to save the data.
        save_to (str): The file to save the results.
        processes (int, optional): The number of processes to use.
            (default: :obj:`1`)
    """

    def __init__(
        self,
        data_dir: str,
        save_to: str,
        processes: int = 1,
    ):
        r"""Initialize the MINT benchmark.

        Args:
            data_dir (str): The directory to save the data.
            save_to (str): The file to save the results.
            processes (int, optional): The number of processes to use for
                parallel processing. (default: :obj:`1`)
        """
        super().__init__("mint", data_dir, save_to, processes)

    def download(self):
        r"""Download the MINT dataset."""
        # 这里可以根据实际情况实现数据下载逻辑
        # 例如从 Hugging Face Hub 或其他数据源下载
        logger.info("MINT dataset download not implemented yet.")
        pass
    
    def _check_task_completed(self, task_id: str) -> bool:
        """检查任务是否已完成"""
        for data in self._results:
            if data["task_id"] == task_id:
                return True
        return False 

    def dump_tasks(self, save_path: str, datas):
        """保存任务数据到文件"""
        constructed_data = []
        for idx, data in enumerate(datas):
            tmp_dict = {
                'idx': idx,
                'task_id': data['id'],
                'prompt': data['prompt'],
                'reference': data['reference']
            }
            constructed_data.append(tmp_dict)
        
        with open(save_path, 'w', encoding="utf-8") as f:
            json.dump(constructed_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Successfully dumped tasks to {save_path}")

    def load(self, force_download=False):
        r"""Load the MINT dataset.

        Args:
            force_download (bool, optional): Whether to force download the data.
        """
        if force_download:
            logger.info("Force downloading data.")
            self.download()

        # 定义数据文件路径
        data_file = self.data_dir / "test.json"
        
        # 检查数据文件是否存在
        if not data_file.exists():
            logger.info("Data not found. Downloading data.")
            self.download()

        # 加载数据
        try:
            with open(data_file, "r", encoding="utf-8") as f:
                raw_data = [json.loads(line) for line in f.readlines()]

            self._data = raw_data
            
            
        except FileNotFoundError:
            logger.warning(f"Data file {data_file} not found. Creating empty dataset.")
            self._data = []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON data: {e}")
            self._data = []
            
        return self
    
    def _load_results_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """从文件加载结果"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                _results = json.load(f)
            return _results
        except Exception as e:
            logger.warning(f"The file {file_path} does not exist: {e}")
            return []
    
    def _save_results_to_file(self, results: List[Dict[str, Any]], file_path: str):
        """保存结果到文件"""
        base_dir = os.path.dirname(file_path)
        os.makedirs(base_dir, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    @property
    def train(self):
        r"""Get the training set."""
        self.load()
        return self._data
    
    def _load_tasks(
        self,
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        r"""Load tasks from the dataset."""
        self.load()
        
        datas = self._data.copy()
        
        if randomize:
            random.shuffle(datas)
        if subset:
            datas = datas[:subset]
        
        if idx is not None and len(idx) > 0:   
            datas = [datas[i] for i in idx if i < len(datas)]
                
        return datas
    
    def get_formal_answer(self, question: str, text: str) -> str:
        """获取格式化的最终答案"""
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        )
        
        agent = ChatAgent(
            "You are a helpful assistant that can answer questions and provide final answers.",
            model=model,
        )
        
        prompt = f"""
I am solving a question:
<question>
{question}
</question>

Now, I have solved the question, the primary answer is as follows:
<answer>
{text}
</answer>

Please extract and format the final answer from the primary answer. The final answer should be:
- Concise and direct
- In the format that best matches the question type
- Without unnecessary explanations or context
- If the answer has multiple items, please use "and" to connect them (e.g., "item1 and item2" for two items, "item1, item2 and item3" for three or more items)
- Do not use commas alone to separate multiple items when there are only two items

Please output only the final answer without any other text.
        """    
        resp = agent.step(prompt)
        return resp.msgs[0].content

    def run_role_playing(
        self,
        user_role_name: str,
        assistant_role_name: str,
        user_agent_kwargs: dict,
        assistant_agent_kwargs: dict,
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        save_result: bool = False,
    ) -> Dict[str, Any]:
        """运行角色扮演模式的基准测试"""

        # 加载任务数据
        datas = self._load_tasks(randomize, subset, idx)
        logger.info(f"Number of tasks: {len(datas)}")

        self._results = []
        
        if save_result:
            self._results = self._load_results_from_file(self.save_to)

        # 处理任务
        for task in tqdm(datas, desc="Running"):
            if self._check_task_completed(task["id"]):
                logger.success(f"Task already completed: {task['id']}")
                continue
                
            try:
                logger.info(f"Task Question: {task['prompt']}")

                task_kwargs = {
                    'task_prompt': task['prompt'],
                    'with_task_specify': False,
                }

                society = OwlGaiaRolePlaying(
                    **task_kwargs,
                    user_role_name=user_role_name,
                    user_agent_kwargs=user_agent_kwargs,
                    assistant_role_name=assistant_role_name,
                    assistant_agent_kwargs=assistant_agent_kwargs,
                )

                raw_answer, chat_history, token_info = run_society(society)
                try:
                    answer = extract_pattern(raw_answer, "final_answer")
                except Exception as e:
                    logger.error(f"Error in extracting final answer from text {raw_answer}: {e}")
                    answer = None

                logger.info(f"Model answer: {answer}, Ground truth: {task['reference']}")

                _result_info = {
                    "task_id": task["id"],
                    "question": task["prompt"],
                    "model_answer": answer,
                    "ground_truth": task["reference"],
                    "score": self.question_scorer(answer, task["reference"]),
                    "token_info": token_info,
                    "history": chat_history,
                }
                self._results.append(_result_info)

            except Exception as e:
                logger.error(f"Error in processing task: {e}")
    
            if save_result:
                self._save_results_to_file(self._results, self.save_to)

        return self._generate_summary()

    def run(
        self,
        agent: ChatAgent,
        max_tries: int = 3,
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        save_result: bool = False,
    ) -> Dict[str, Any]:
        r"""Run the benchmark with a single agent."""
        
        datas = self._load_tasks(randomize, subset, idx)
        
        self._results = []
        
        if save_result:
            self._results = self._load_results_from_file(self.save_to)

        for task in tqdm(datas, desc="Running"):
            if self._check_task_completed(task["id"]):
                logger.success(f"Task already completed: {task['id']}")
                continue
            
            success = False
            tries = 0
            trajectory_with_retry: List[dict] = []
            
            while not success and tries < max_tries:
                tries += 1
                logger.info(f"Attempt {tries}/{max_tries} for task {task['id']}")
                
                try:
                    logger.info(f"Task Question: {task['prompt']}")
                    agent.reset()
                    
                    prompt = task['prompt']
                    
                    resp = agent.step(prompt)
                    raw_answer = resp.msgs[0].content
                    answer = self.get_formal_answer(task['prompt'], raw_answer)
                    
                    logger.info(f"Model answer: {answer}, Ground truth: {task['reference']}")

                    score = self.question_scorer(answer, task["reference"])
                    success = score == True
                    trajectory_dict = {
                        "attempts": tries,
                        "model_answer": answer,
                        "ground_truth": task["reference"],
                        "success": success,
                        "trajectory": agent.chat_history
                    }
                    trajectory_with_retry.append(trajectory_dict)
                    
                    if success or tries == max_tries:
                        _result_info = {
                            "task_id": task["id"],
                            "question": task["prompt"],
                            "model_answer": answer,
                            "ground_truth": task["reference"],
                            "score": score,
                            "attempts": tries,  
                            "trajectory": trajectory_with_retry
                        }
                        self._results.append(_result_info)

                except Exception as e:
                    logger.error(f"Error in processing task: {e}")  
                    
            if save_result:
                self._save_results_to_file(self._results, self.save_to)
                
        return self._generate_summary()
    
    def run_workforce_with_retry(
        self,
        workforce: Workforce,  
        max_tries: int = 3,
        max_replanning_tries: int = 2,
        randomize: bool = False,
        subset: Optional[int] = None,
        idx: Optional[List[int]] = None,
        save_result: bool = False,
    ) -> Dict[str, Any]:
        r"""Run the benchmark with workforce and retry mechanism."""
        
        tasks = self._load_tasks(randomize, subset, idx)
        self._results = []
        
        if save_result:
            self._results = self._load_results_from_file(self.save_to)
        
        for task in tqdm(tasks, desc=f"Running tasks"):
            if self._check_task_completed(task["id"]):
                logger.success(f"Task already completed: {task['id']}")
                continue

            success = False
            tries = 0
            trajectory_with_retry: List[dict] = []
            
            while not success and tries < max_tries:
                tries += 1
                logger.info(f"Attempt {tries}/{max_tries} for task {task['id']}")
                
                try:
                    logger.info(f"Task Question: {task['prompt']}")
                    camel_task = self._create_task(task)
                    if workforce.is_running():
                        workforce.stop()
                    processed_task = workforce.process_task(camel_task, max_replanning_tries=max_replanning_tries)

                    try:
                        answer_prompt = """
I am solving a question:
<question>
{question}
</question>

Now, I have solved the question by decomposing it into several subtasks, the subtask information is as follows:
<subtask_info>
{subtask_info}
</subtask_info>

Please extract and format the final answer from the primary answer. The final answer should be:
- Concise and direct
- In the format that best matches the question type
- Without unnecessary explanations or context
- If the answer has multiple items, please use "and" to connect them (e.g., "item1 and item2" for two items, "item1, item2 and item3" for three or more items)
- Do not use commas alone to separate multiple items when there are only two items

Please output only the final answer without any other text.
"""    
                        answer = workforce.get_workforce_final_answer(processed_task, prompt=answer_prompt)
                    except Exception as e:
                        logger.error(f"Error extracting final answer: {e}")
                        answer = None

                    logger.info(f"Model answer: {answer}, Ground truth: {task['reference']}")
                    
                    score = self.question_scorer(answer, task["reference"])
                    logger.info(f"Score: {score}")
                    success = score == True
                    trajectory_dict = {
                        "attempts": tries,
                        "model_answer": answer,
                        "ground_truth": task["reference"],
                        "success": success,
                        "trajectory": workforce.get_overall_task_solve_trajectory()
                    }
                    trajectory_with_retry.append(trajectory_dict)
                    
                    if success or tries == max_tries:
                        _result_info = {
                            "task_id": task["id"],
                            "question": task["prompt"],
                            "model_answer": answer,
                            "ground_truth": task["reference"],
                            "score": score,
                            "attempts": tries,
                            "trajectory": trajectory_with_retry
                        }
                        self._results.append(_result_info)

                except Exception as e:
                    logger.error(f"Error in processing task (attempt {tries}): {e}")
                    if tries == max_tries:
                        _result_info = {
                            "task_id": task["id"],
                            "question": task["prompt"],
                            "model_answer": None,
                            "ground_truth": task["reference"],
                            "score": False,
                            "attempts": tries,
                            "trajectory": trajectory_with_retry
                        }
                        self._results.append(_result_info)

            if save_result:
                self._save_results_to_file(self._results, self.save_to)

        return self._generate_summary()
        
    def _create_task(self, task: Dict[str, Any]) -> Task:
        r"""Create a Task object from task data.

        Args:
            task (Dict[str, Any]): The task data containing id and prompt.

        Returns:
            Task: The CAMEL Task object.
        """
        return Task(id=str(task["id"]), content=task["prompt"])

    def _generate_summary(self) -> Dict[str, Any]:
        r"""Generate and return a summary of the benchmark results."""
        correct = sum(1 for result in self._results if result["score"])
        return {
            "total": len(self._results),
            "correct": correct,
            "results": self._results,
            "accuracy": correct / len(self._results) if len(self._results) > 0 else 0,
        }

    def question_scorer(self, model_answer: str, ground_truth: str) -> bool:
        r"""Scorer for the MINT benchmark.

        Args:
            model_answer (str): The model answer.
            ground_truth (str): The ground truth answer.

        Returns:
            bool: Whether the model answer is correct.
        """
        if model_answer is None or ground_truth is None:
            return False

        def is_float(element: Any) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False

        # 如果答案是数字
        if is_float(ground_truth):
            logger.info(f"Evaluating {model_answer} as a number.")
            normalized_answer = self.normalize_number_str(model_answer)
            return abs(normalized_answer - float(ground_truth)) < 1e-6

        # 如果答案包含逗号或分号，作为列表处理
        elif any(char in ground_truth for char in [",", ";"]):
            logger.info(f"Evaluating {model_answer} as a comma separated list.")
            gt_elems = self.split_string(ground_truth)
            ma_elems = self.split_string(model_answer)

            if len(gt_elems) != len(ma_elems):
                logger.warning("Answer lists have different lengths, returning False.")
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = self.normalize_number_str(ma_elem)
                    comparisons.append(abs(normalized_ma_elem - float(gt_elem)) < 1e-6)
                else:
                    ma_elem = self.normalize_str(ma_elem, remove_punct=False)
                    gt_elem = self.normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)
        else:
            # 作为字符串处理
            logger.info(f"Evaluating {model_answer} as a string.")
            ma_elem = self.normalize_str(model_answer)
            gt_elem = self.normalize_str(ground_truth)
            return ma_elem == gt_elem

    def normalize_number_str(self, number_str: str) -> float:
        """标准化数字字符串"""
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            logger.error(f"String {number_str} cannot be normalized to number str.")
            return float("inf")

    def split_string(self, s: str, char_list: Optional[List[str]] = None) -> List[str]:
        r"""Split a string based on a list of characters.

        Args:
            s (str): The string to split.
            char_list (Optional[List[str]], optional): The list of characters to split on.
        """
        if char_list is None:
            char_list = [",", ";"]
        pattern = f"[{''.join(char_list)}]"
        return [item.strip() for item in re.split(pattern, s)]

    def normalize_str(self, input_str: str, remove_punct: bool = True) -> str:
        r"""Normalize a string.

        Args:
            input_str: The input string to normalize.
            remove_punct: Whether to remove punctuation.

        Returns:
            str: The normalized string.
        """
        if input_str is None:
            return ""
        
        no_spaces = re.sub(r"\s", "", input_str)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower() 