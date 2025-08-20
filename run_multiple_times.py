#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顺序运行30次run_lenovo_workforce.py的脚本
每次运行都会保存独立的日志和结果文件，并统计运行时间
"""

import os
import sys
import time
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_multiple_times.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MultipleRunsExecutor:
    def __init__(self, script_path: str = "run_lenovo_workforce.py", num_runs: int = 30):
        """
        初始化多次运行执行器
        
        Args:
            script_path: 要运行的脚本路径
            num_runs: 运行次数
        """
        self.script_path = script_path
        self.num_runs = num_runs
        self.base_output_dir = Path("multiple_runs_output")
        self.results_summary = []
        
        # 创建输出目录
        self.base_output_dir.mkdir(exist_ok=True)
        
    def create_run_directory(self, run_number: int) -> Path:
        """为每次运行创建独立的目录"""
        run_dir = self.base_output_dir / f"run_{run_number:03d}"
        run_dir.mkdir(exist_ok=True)
        return run_dir
    
    def backup_existing_files(self, run_dir: Path):
        """备份可能被覆盖的文件"""
        # 备份现有的结果文件
        existing_files = [
            "result.md",
            "results/workforce/workforce_1_pass1_gpt4o.json",
            "logs/",
            "tmp/"
        ]
        
        for file_path in existing_files:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    backup_path = run_dir / f"backup_{Path(file_path).name}"
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.copytree(file_path, backup_path)
                else:
                    backup_path = run_dir / f"backup_{Path(file_path).name}"
                    shutil.copy2(file_path, backup_path)
    
    def run_single_execution(self, run_number: int) -> Dict[str, Any]:
        """运行单次执行"""
        run_dir = self.create_run_directory(run_number)
        start_time = time.time()
        
        logger.info(f"开始第 {run_number} 次运行...")
        
        # 备份现有文件
        self.backup_existing_files(run_dir)
        
        # 设置环境变量，确保日志输出到指定目录
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
        
        try:
            # 运行脚本，实时显示输出
            process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',
                bufsize=1,
                universal_newlines=True
            )
            
            # 实时读取输出并显示
            stdout_lines = []
            stderr_lines = []
            
            # 参考IntelligentSenseService.py的实现方式
            while True:
                if process.stdout:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        stdout_lines.append(output)
                        try:
                            print(f"[运行{run_number}] {output.rstrip()}")
                        except UnicodeDecodeError:
                            # 处理编码问题
                            try:
                                decoded_output = output.encode('latin-1').decode('utf-8', errors='replace')
                                print(f"[运行{run_number}] {decoded_output.rstrip()}")
                            except:
                                print(f"[运行{run_number}] [编码错误] {repr(output)}")
                else:
                    break
            
            # 获取剩余输出
            try:
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout_lines.append(remaining_stdout)
                    for line in remaining_stdout.splitlines():
                        if line.strip():
                            try:
                                print(f"[运行{run_number}] {line}")
                            except UnicodeDecodeError:
                                try:
                                    decoded_line = line.encode('latin-1').decode('utf-8', errors='replace')
                                    print(f"[运行{run_number}] {decoded_line}")
                                except:
                                    print(f"[运行{run_number}] [编码错误] {repr(line)}")
                if remaining_stderr:
                    stderr_lines.append(remaining_stderr)
                    for line in remaining_stderr.splitlines():
                        if line.strip():
                            try:
                                print(f"[运行{run_number}-错误] {line}")
                            except UnicodeDecodeError:
                                try:
                                    decoded_line = line.encode('latin-1').decode('utf-8', errors='replace')
                                    print(f"[运行{run_number}-错误] {decoded_line}")
                                except:
                                    print(f"[运行{run_number}-错误] [编码错误] {repr(line)}")
            except Exception as e:
                print(f"[运行{run_number}-错误] 读取剩余输出时出错: {e}")
            
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 保存输出
            with open(run_dir / "stdout.log", "w", encoding="utf-8") as f:
                f.write(stdout)
            
            with open(run_dir / "stderr.log", "w", encoding="utf-8") as f:
                f.write(stderr)
            
            # 复制运行期间生成的文件
            self.copy_generated_files(run_dir)
            
            # 记录运行结果
            run_result = {
                "run_number": run_number,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "execution_time_seconds": execution_time,
                "execution_time_minutes": execution_time / 60,
                "return_code": process.returncode,
                "success": process.returncode == 0,
                "stdout_lines": len(stdout.splitlines()),
                "stderr_lines": len(stderr.splitlines()),
                "output_dir": str(run_dir)
            }
            
            if process.returncode == 0:
                logger.info(f"第 {run_number} 次运行成功完成，耗时: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")
            else:
                logger.error(f"第 {run_number} 次运行失败，返回码: {process.returncode}")
                logger.error(f"错误输出: {stderr}")
            
            return run_result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            error_result = {
                "run_number": run_number,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "execution_time_seconds": execution_time,
                "execution_time_minutes": execution_time / 60,
                "return_code": -1,
                "success": False,
                "error": str(e),
                "output_dir": str(run_dir)
            }
            
            logger.error(f"第 {run_number} 次运行发生异常: {e}")
            return error_result
    
    def copy_generated_files(self, run_dir: Path):
        """复制运行期间生成的文件到运行目录"""
        files_to_copy = [
            "result.md",
            "results/workforce/workforce_1_pass1_gpt4o.json",
            "overall_task_solve_trajectory.json"
        ]
        
        for file_path in files_to_copy:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    dest_path = run_dir / Path(file_path).name
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(file_path, dest_path)
                else:
                    dest_path = run_dir / Path(file_path).name
                    shutil.copy2(file_path, dest_path)
        
        # 复制日志目录
        if os.path.exists("logs"):
            logs_dest = run_dir / "logs"
            if logs_dest.exists():
                shutil.rmtree(logs_dest)
            shutil.copytree("logs", logs_dest)
        
        # 复制临时目录（如果存在）
        if os.path.exists("tmp"):
            tmp_dest = run_dir / "tmp"
            if tmp_dest.exists():
                shutil.rmtree(tmp_dest)
            shutil.copytree("tmp", tmp_dest)
    
    def run_all_executions(self):
        """运行所有执行"""
        logger.info(f"开始顺序运行 {self.num_runs} 次 {self.script_path}")
        logger.info(f"输出目录: {self.base_output_dir}")
        
        total_start_time = time.time()
        
        for run_number in range(1, self.num_runs + 1):
            run_result = self.run_single_execution(run_number)
            self.results_summary.append(run_result)
            
            # 每5次运行保存一次中间结果
            if run_number % 5 == 0:
                self.save_intermediate_results()
        
        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time
        
        # 保存最终结果
        self.save_final_results(total_execution_time)
        
        # 生成统计报告
        self.generate_statistics_report()
        
        logger.info(f"所有 {self.num_runs} 次运行完成！")
        logger.info(f"总耗时: {total_execution_time:.2f} 秒 ({total_execution_time/60:.2f} 分钟)")
    
    def save_intermediate_results(self):
        """保存中间结果"""
        intermediate_file = self.base_output_dir / "intermediate_results.json"
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(self.results_summary, f, indent=2, ensure_ascii=False)
    
    def save_final_results(self, total_execution_time: float):
        """保存最终结果"""
        final_results = {
            "summary": {
                "total_runs": self.num_runs,
                "total_execution_time_seconds": total_execution_time,
                "total_execution_time_minutes": total_execution_time / 60,
                "total_execution_time_hours": total_execution_time / 3600,
                "average_execution_time_seconds": total_execution_time / self.num_runs,
                "successful_runs": sum(1 for r in self.results_summary if r["success"]),
                "failed_runs": sum(1 for r in self.results_summary if not r["success"])
            },
            "runs": self.results_summary
        }
        
        final_file = self.base_output_dir / "final_results.json"
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    def generate_statistics_report(self):
        """生成统计报告"""
        successful_runs = [r for r in self.results_summary if r["success"]]
        failed_runs = [r for r in self.results_summary if not r["success"]]
        
        if successful_runs:
            execution_times = [r["execution_time_seconds"] for r in successful_runs]
            min_time = min(execution_times)
            max_time = max(execution_times)
            avg_time = sum(execution_times) / len(execution_times)
            
            report = f"""
=== 多次运行统计报告 ===

总运行次数: {self.num_runs}
成功运行次数: {len(successful_runs)}
失败运行次数: {len(failed_runs)}
成功率: {len(successful_runs)/self.num_runs*100:.2f}%

执行时间统计 (仅成功运行):
- 最短时间: {min_time:.2f} 秒 ({min_time/60:.2f} 分钟)
- 最长时间: {max_time:.2f} 秒 ({max_time/60:.2f} 分钟)
- 平均时间: {avg_time:.2f} 秒 ({avg_time/60:.2f} 分钟)

总执行时间: {sum(r["execution_time_seconds"] for r in self.results_summary):.2f} 秒 ({sum(r["execution_time_seconds"] for r in self.results_summary)/60:.2f} 分钟)

输出目录: {self.base_output_dir}
详细结果文件: {self.base_output_dir}/final_results.json
"""
        else:
            report = f"""
=== 多次运行统计报告 ===

总运行次数: {self.num_runs}
成功运行次数: 0
失败运行次数: {len(failed_runs)}
成功率: 0.00%

所有运行都失败了，请检查脚本和配置。

输出目录: {self.base_output_dir}
详细结果文件: {self.base_output_dir}/final_results.json
"""
        
        # 保存报告
        report_file = self.base_output_dir / "statistics_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        # 打印报告
        print(report)
        logger.info("统计报告已生成")


def main():
    """主函数"""
    # 检查脚本文件是否存在
    script_path = "run_lenovo_workforce.py"
    if not os.path.exists(script_path):
        logger.error(f"脚本文件 {script_path} 不存在！")
        sys.exit(1)
    
    # 创建执行器并运行
    executor = MultipleRunsExecutor(script_path=script_path, num_runs=30)
    executor.run_all_executions()


if __name__ == "__main__":
    main() 