#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定版：顺序运行30次run_lenovo_workforce.py的脚本
完全参考IntelligentSenseService.py的实现方式
"""

import os
import sys
import time
import json
import shutil
import subprocess
import threading
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
        print(f"\n{'='*50}")
        print(f"=== 第 {run_num} 次运行 ===")
        print(f"{'='*50}")
        
        # 创建本次运行的目录
        run_dir = output_dir / f"run_{run_num:03d}"
        run_dir.mkdir(exist_ok=True)
        
        # 记录开始时间
        start_time = time.time()
        start_datetime = datetime.now()
        
        try:
            # 运行脚本
            print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"正在运行: python run_lenovo_workforce.py")
            print("-" * 50)
            
            # 参考IntelligentSenseService.py的实现方式
            python_exe = sys.executable
            cmd = [python_exe, "run_lenovo_workforce.py"]
            
            # 创建进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                text=True,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',  # 处理编码错误
                env=dict(os.environ, PYTHONIOENCODING='utf-8')  # 设置Python IO编码
            )
            
            # 实时读取输出
            stdout_lines = []
            
            def log_output():
                try:
                    # 为每次运行创建独立的日志文件
                    log_file = run_dir / f"run_{run_num:03d}_output.log"
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write(f"脚本执行开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"执行命令: {' '.join(cmd)}\n")
                        f.write(f"进程ID: {process.pid}\n")
                        f.write("=" * 50 + "\n")
                        f.flush()
                        
                        # 实时读取并写入日志
                        if process.stdout:
                            while True:
                                output = process.stdout.readline()
                                if output == '' and process.poll() is not None:
                                    break
                                if output:
                                    # 处理编码问题
                                    try:
                                        # 尝试直接写入
                                        f.write(output)
                                        f.flush()
                                        stdout_lines.append(output)
                                        # 同时输出到控制台
                                        print(f"[运行{run_num}] {output.rstrip()}")
                                    except UnicodeDecodeError:
                                        # 如果出现编码错误，尝试重新编码
                                        try:
                                            decoded_output = output.encode('latin-1').decode('utf-8', errors='replace')
                                            f.write(decoded_output)
                                            f.flush()
                                            stdout_lines.append(decoded_output)
                                            print(f"[运行{run_num}] {decoded_output.rstrip()}")
                                        except:
                                            # 最后的回退方案
                                            safe_output = output.encode('utf-8', errors='replace').decode('utf-8')
                                            f.write(safe_output)
                                            f.flush()
                                            stdout_lines.append(safe_output)
                                            print(f"[运行{run_num}] {safe_output.rstrip()}")
                        else:
                            f.write("无法捕获脚本输出\n")
                            print(f"[运行{run_num}] 无法捕获脚本输出")
                        
                        # 获取返回码
                        return_code = process.poll()
                        f.write("=" * 50 + "\n")
                        f.write(f"脚本执行结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"返回码: {return_code}\n")
                        
                except Exception as e:
                    print(f"[运行{run_num}-错误] 写入日志文件时发生错误: {e}")
            
            # 启动日志处理线程
            log_thread = threading.Thread(target=log_output, daemon=True)
            log_thread.start()
            
            # 等待进程完成
            process.wait()
            
            # 等待日志线程完成
            log_thread.join(timeout=5)
            
            # 记录结束时间
            end_time = time.time()
            end_datetime = datetime.now()
            execution_time = end_time - start_time
            
            # 保存输出到文件
            stdout = ''.join(stdout_lines)
            with open(run_dir / f"run_{run_num:03d}_stdout.txt", "w", encoding="utf-8") as f:
                f.write(stdout)
            
            # 复制生成的文件，为每次运行创建独立的文件
            files_to_copy = [
                "result.md",
                "results/workforce/workforce_1_pass1_gpt4o.json",
                "overall_task_solve_trajectory.json"
            ]
            
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    # 为每次运行创建独立的文件名
                    file_name = Path(file_path).name
                    name_without_ext = Path(file_name).stem
                    ext = Path(file_name).suffix
                    dest_path = run_dir / f"run_{run_num:03d}_{name_without_ext}{ext}"
                    shutil.copy2(file_path, dest_path)
            
            # 复制日志和临时目录，为每次运行创建独立的目录
            if os.path.exists("logs"):
                logs_dest = run_dir / f"run_{run_num:03d}_logs"
                if logs_dest.exists():
                    shutil.rmtree(logs_dest)
                shutil.copytree("logs", logs_dest)
            if os.path.exists("tmp"):
                tmp_dest = run_dir / f"run_{run_num:03d}_tmp"
                if tmp_dest.exists():
                    shutil.rmtree(tmp_dest)
                shutil.copytree("tmp", tmp_dest)
            
            # 记录结果
            run_result = {
                "run_number": run_num,
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat(),
                "execution_time_seconds": execution_time,
                "execution_time_minutes": execution_time / 60,
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout_lines": len(stdout.splitlines()),
                "process_id": process.pid
            }
            
            all_results.append(run_result)
            
            print("-" * 50)
            if process.returncode == 0:
                print(f"✅ 成功完成！耗时: {execution_time:.2f} 秒 ({execution_time/60:.2f} 分钟)")
            else:
                print(f"❌ 运行失败！返回码: {process.returncode}")
            
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
    
    print(f"\n{'='*60}")
    print(f"=== 运行完成 ===")
    print(f"{'='*60}")
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
    
    # 为每次运行创建独立的summary文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {summary_file}")

if __name__ == "__main__":
    # 可以通过命令行参数指定运行次数
    num_runs = 10
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print("使用默认运行次数: 30")
    
    run_multiple_times(num_runs) 