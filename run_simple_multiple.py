#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版：顺序运行30次run_lenovo_workforce.py的脚本
"""

import os
import sys
import time
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

def run_multiple_times(num_runs=30):
    """顺序运行指定次数的脚本"""
    
    # 创建输出目录
    output_dir = Path("multiple_runs_output")
    output_dir.mkdir(exist_ok=True)
    
    # 记录所有运行结果
    all_results = []
    total_start_time = time.time()
    
    print(f"开始顺序运行 {num_runs} 次 run_lenovo_workforce.py")
    print(f"输出目录: {output_dir}")
    
    for run_num in range(1, num_runs + 1):
        print(f"\n=== 第 {run_num} 次运行 ===")
        
        # 创建本次运行的目录
        run_dir = output_dir / f"run_{run_num:03d}"
        run_dir.mkdir(exist_ok=True)
        
        # 记录开始时间
        start_time = time.time()
        start_datetime = datetime.now()
        
        try:
            # 运行脚本
            print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 使用Popen来实时显示输出
            process = subprocess.Popen(
                [sys.executable, "run_lenovo_workforce.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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
                            print(f"[运行{run_num}] {output.rstrip()}")
                        except UnicodeDecodeError:
                            # 处理编码问题
                            try:
                                decoded_output = output.encode('latin-1').decode('utf-8', errors='replace')
                                print(f"[运行{run_num}] {decoded_output.rstrip()}")
                            except:
                                print(f"[运行{run_num}] [编码错误] {repr(output)}")
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
                                print(f"[运行{run_num}] {line}")
                            except UnicodeDecodeError:
                                try:
                                    decoded_line = line.encode('latin-1').decode('utf-8', errors='replace')
                                    print(f"[运行{run_num}] {decoded_line}")
                                except:
                                    print(f"[运行{run_num}] [编码错误] {repr(line)}")
                if remaining_stderr:
                    stderr_lines.append(remaining_stderr)
                    for line in remaining_stderr.splitlines():
                        if line.strip():
                            try:
                                print(f"[运行{run_num}-错误] {line}")
                            except UnicodeDecodeError:
                                try:
                                    decoded_line = line.encode('latin-1').decode('utf-8', errors='replace')
                                    print(f"[运行{run_num}-错误] {decoded_line}")
                                except:
                                    print(f"[运行{run_num}-错误] [编码错误] {repr(line)}")
            except Exception as e:
                print(f"[运行{run_num}-错误] 读取剩余输出时出错: {e}")
            
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            
            # 创建结果对象
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            result = Result(process.returncode, stdout, stderr)
            
            # 记录结束时间
            end_time = time.time()
            end_datetime = datetime.now()
            execution_time = end_time - start_time
            
            # 保存输出
            with open(run_dir / "stdout.txt", "w", encoding="utf-8") as f:
                f.write(result.stdout)
            
            with open(run_dir / "stderr.txt", "w", encoding="utf-8") as f:
                f.write(result.stderr)
            
            # 复制生成的文件
            files_to_copy = [
                "result.md",
                "results/workforce/workforce_1_pass1_gpt4o.json",
                "overall_task_solve_trajectory.json"
            ]
            
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    dest_path = run_dir / Path(file_path).name
                    shutil.copy2(file_path, dest_path)
            
            # 复制日志和临时目录
            if os.path.exists("logs"):
                shutil.copytree("logs", run_dir / "logs", dirs_exist_ok=True)
            if os.path.exists("tmp"):
                shutil.copytree("tmp", run_dir / "tmp", dirs_exist_ok=True)
            
            # 记录结果
            run_result = {
                "run_number": run_num,
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat(),
                "execution_time_seconds": execution_time,
                "execution_time_minutes": execution_time / 60,
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout_lines": len(result.stdout.splitlines()),
                "stderr_lines": len(result.stderr.splitlines())
            }
            
            all_results.append(run_result)
            
            if result.returncode == 0:
                print(f"✅ 成功完成！耗时: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")
            else:
                print(f"❌ 运行失败！返回码: {result.returncode}")
                print(f"错误信息: {result.stderr[:200]}...")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"❌ 运行异常: {e}")
            
            run_result = {
                "run_number": run_num,
                "start_time": start_datetime.isoformat(),
                "end_time": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "execution_time_minutes": execution_time / 60,
                "success": False,
                "return_code": -1,
                "error": str(e)
            }
            
            all_results.append(run_result)
    
    # 计算总时间
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # 生成统计信息
    successful_runs = [r for r in all_results if r["success"]]
    failed_runs = [r for r in all_results if not r["success"]]
    
    print(f"\n=== 运行完成 ===")
    print(f"总运行次数: {num_runs}")
    print(f"成功次数: {len(successful_runs)}")
    print(f"失败次数: {len(failed_runs)}")
    print(f"成功率: {len(successful_runs)/num_runs*100:.2f}%")
    print(f"总耗时: {total_execution_time:.2f} 秒 ({total_execution_time/60:.2f} 分钟)")
    
    if successful_runs:
        times = [r["execution_time_seconds"] for r in successful_runs]
        print(f"平均耗时: {sum(times)/len(times):.2f} 秒")
        print(f"最短耗时: {min(times):.2f} 秒")
        print(f"最长耗时: {max(times):.2f} 秒")
    
    # 保存详细结果
    summary = {
        "total_runs": num_runs,
        "successful_runs": len(successful_runs),
        "failed_runs": len(failed_runs),
        "success_rate": len(successful_runs)/num_runs*100,
        "total_execution_time_seconds": total_execution_time,
        "total_execution_time_minutes": total_execution_time / 60,
        "runs": all_results
    }
    
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_dir}/summary.json")

if __name__ == "__main__":
    # 可以通过命令行参数指定运行次数
    num_runs = 30
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print("使用默认运行次数: 30")
    
    run_multiple_times(num_runs) 