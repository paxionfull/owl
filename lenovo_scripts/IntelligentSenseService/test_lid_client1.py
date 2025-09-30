import win32file
import win32pipe
import pywintypes
import time
import logging
import os
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_test_file(file_path: str = "hello_lenovo.txt") -> bool:
    """
    创建测试文件
    
    Args:
        file_path: 要创建的文件路径
    
    Returns:
        创建是否成功
    """
    try:
        # 在当前目录创建空文件
        with open(file_path, "w", encoding='utf-8') as f:
            f.write("")  # 创建空文件
        logging.info(f"✅ 成功创建文件: {file_path}")
        return True
    except Exception as e:
        logging.error(f"❌ 创建文件失败: {e}")
        return False

def delete_test_file(file_path: str = "hello_lenovo.txt") -> bool:
    """
    删除测试文件
    
    Args:
        file_path: 要删除的文件路径
    
    Returns:
        删除是否成功
    """
    try:
        # 规范化路径
        path = Path(file_path).resolve()
        
        # 安全检查
        if not path.exists():
            logging.warning(f"文件不存在: {path}")
            return False
        
        if path.is_dir():
            logging.error(f"路径指向目录而非文件: {path}")
            return False
        
        # 尝试删除文件
        os.remove(path)
        logging.info(f"🗑️ 文件已成功删除: {path}")
        return True
        
    except PermissionError:
        logging.error(f"权限不足，无法删除文件: {file_path}")
    except FileNotFoundError:
        logging.warning(f"文件在删除前已被移除: {file_path}")
    except Exception as e:
        logging.error(f"删除文件时发生错误: {e}")
    
    return False

def send_message_to_pipe(
    pipe_name: str = r'\\.\pipe\LidModeStatus',
    message: int | str | bytes = "Hello from client!",
    max_attempts: int = 5,
    retry_interval: int = 1,
    timeout: int = 10
) -> bytes:
    """
    向命名管道发送消息并接收响应
    参考IntelligentSenseService.py的实现方式
    
    Args:
        pipe_name: 管道名称
        message: 消息内容（支持int/str/bytes类型）
        max_attempts: 最大连接尝试次数
        retry_interval: 重试间隔（秒）
        timeout: 超时时间（秒）
    
    Returns:
        服务端响应的原始字节数据
    """
    pipe = None
    response = b""
    
    try:
        # 处理不同类型的消息
        if isinstance(message, bytes):
            # 直接使用bytes类型（作为十六进制数据）
            message_bytes = message
            logging.info(f"准备发送十六进制消息: {message_bytes.hex().upper()}")
        elif isinstance(message, str):
            # 字符串按UTF-8编码处理
            message_bytes = message.encode('utf-8')
            logging.info(f"准备发送字符串消息: {message}")
        elif isinstance(message, int):
            # 整数转单字节（确保在0-255范围内）
            if 0 <= message <= 255:
                message_bytes = bytes([message])
            else:
                raise ValueError(f"整数超出单字节范围: {message}")
            logging.info(f"准备发送单字节消息: 0x{message:02X}")
        else:
            raise TypeError(f"不支持的消息类型: {type(message).__name__}")
        
        # 连接到管道
        for attempt in range(max_attempts):
            try:
                pipe = win32file.CreateFile(
                    pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None
                )
                logging.info(f"成功连接到管道: {pipe_name}")
                break
            except pywintypes.error as e:
                if e.args[0] == 2:  # 管道不存在
                    if attempt < max_attempts - 1:
                        logging.info(f"管道未找到，{retry_interval}秒后重试... (尝试 {attempt + 1}/{max_attempts})")
                        time.sleep(retry_interval)
                    else:
                        raise
                else:
                    raise
        
        # 设置管道为消息模式
        win32pipe.SetNamedPipeHandleState(  # type: ignore
            pipe,
            win32pipe.PIPE_READMODE_MESSAGE,
            None,
            None
        )
        
        # 发送消息
        win32file.WriteFile(pipe, message_bytes)  # type: ignore
        logging.info(f"已发送消息（{len(message_bytes)}字节）")
        
        # 接收响应
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                hr, data = win32file.ReadFile(pipe, 65536)  # type: ignore
                if isinstance(data, bytes):
                    response += data
                    if len(data) < 65536:  # 消息结束
                        break
            except pywintypes.error as e:
                if e.args[0] == 232:  # ERROR_MORE_DATA
                    continue
                else:
                    raise
        
        logging.info(f"收到响应（{len(response)}字节）: {response.hex().upper()}")
        
    except Exception as e:
        logging.error(f"管道通信错误: {str(e)}")
        raise
    finally:
        if pipe:
            win32file.CloseHandle(pipe)  # type: ignore
    
    return response

def test_lid_closed_scenario():
    """测试盖子合上场景"""
    logging.info("🔒 开始测试盖子合上场景")
    
    try:
        # 发送盖子合上命令
        response = send_message_to_pipe(message=0xFD)
        
        # 检查响应
        if response == b'\xFD':
            logging.info("🎯 检测到盖子合上状态确认!")
            # 创建测试文件
            if create_test_file():
                logging.info("✅ 盖子合上测试场景完成")
                return True
            else:
                logging.error("❌ 文件创建失败")
                return False
        else:
            logging.warning(f"⚠️ 未识别的响应: {response.hex().upper()}")
            return False
            
    except Exception as e:
        logging.error(f"❌ 盖子合上测试失败: {e}")
        return False

def test_lid_opened_scenario():
    """测试盖子打开场景"""
    logging.info("🔓 开始测试盖子打开场景")
    
    try:
        # 发送盖子打开命令
        response = send_message_to_pipe(message=0xFE)
        
        # 检查响应
        if response == b'\xFE':
            logging.info("🎯 检测到盖子打开状态确认!")
            # 删除测试文件
            if delete_test_file():
                logging.info("✅ 盖子打开测试场景完成")
                return True
            else:
                logging.warning("⚠️ 文件删除失败或文件不存在")
                return True  # 删除失败不算测试失败
        else:
            logging.warning(f"⚠️ 未识别的响应: {response.hex().upper()}")
            return False
            
    except Exception as e:
        logging.error(f"❌ 盖子打开测试失败: {e}")
        return False

def test_string_message_scenario():
    """测试字符串消息场景"""
    logging.info("💬 开始测试字符串消息场景")
    
    try:
        # 测试hello消息
        response = send_message_to_pipe(message="hello")
        
        try:
            response_str = response.decode('utf-8')
            logging.info(f"📥 收到字符串响应: {response_str}")
            
            if "hello" in response_str.lower():
                logging.info("✅ 字符串消息测试完成")
                return True
            else:
                logging.warning("⚠️ 响应内容未包含预期的hello回复")
                return False
        except UnicodeDecodeError:
            logging.info(f"📥 收到十六进制响应: {response.hex().upper()}")
            return True
            
    except Exception as e:
        logging.error(f"❌ 字符串消息测试失败: {e}")
        return False

def test_time_request_scenario():
    """测试时间请求场景"""
    logging.info("⏰ 开始测试时间请求场景")
    
    try:
        # 请求时间
        response = send_message_to_pipe(message="time")
        
        try:
            response_str = response.decode('utf-8')
            logging.info(f"📥 收到时间响应: {response_str}")
            logging.info("✅ 时间请求测试完成")
            return True
        except UnicodeDecodeError:
            logging.info(f"📥 收到十六进制响应: {response.hex().upper()}")
            return True
            
    except Exception as e:
        logging.error(f"❌ 时间请求测试失败: {e}")
        return False

def run_full_test_suite():
    """运行完整测试套件"""
    logging.info("🔄 开始运行完整测试套件")
    
    test_results = []
    
    # 测试场景列表
    test_scenarios = [
        ("盖子合上检测", test_lid_closed_scenario),
        ("盖子打开检测", test_lid_opened_scenario),
        ("字符串消息", test_string_message_scenario),
        ("时间请求", test_time_request_scenario),
    ]
    
    # 执行所有测试
    for test_name, test_func in test_scenarios:
        logging.info(f"\n{'='*50}")
        logging.info(f"🧪 执行测试: {test_name}")
        logging.info(f"{'='*50}")
        
        try:
            result = test_func()
            test_results.append((test_name, result))
            
            if result:
                logging.info(f"✅ 测试通过: {test_name}")
            else:
                logging.error(f"❌ 测试失败: {test_name}")
                
        except Exception as e:
            logging.error(f"💥 测试异常: {test_name} - {e}")
            test_results.append((test_name, False))
        
        # 测试间隔
        time.sleep(1)
    
    # 输出测试结果摘要
    logging.info(f"\n{'='*50}")
    logging.info("📊 测试结果摘要")
    logging.info(f"{'='*50}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logging.info(f"{status} - {test_name}")
    
    logging.info(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    return passed == total

def monitor_lid_status_continuously(interval: int = 5):
    """持续监控盖子状态"""
    logging.info(f"🔄 启动持续监控模式 (每{interval}秒检查一次，Ctrl+C停止)")
    
    try:
        while True:
            current_time = time.strftime('%H:%M:%S')
            logging.info(f"\n⏰ {current_time} - 执行状态检查...")
            
            # 只测试盖子状态相关功能
            test_lid_closed_scenario()
            time.sleep(1)
            test_lid_opened_scenario()
            
            logging.info(f"😴 等待{interval}秒后继续监控...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logging.info("\n⏹️ 用户中断，监控已停止")
    except Exception as e:
        logging.error(f"💥 监控过程中发生错误: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 联想笔记本盖子状态测试客户端 v2.0")
    print("=" * 60)
    print("基于 IntelligentSenseService.py 风格重写")
    print("=" * 60)
    
    while True:
        print("\n请选择测试模式:")
        print("1. 🧪 运行完整测试套件")
        print("2. 🔒 仅测试盖子合上")
        print("3. 🔓 仅测试盖子打开")
        print("4. 💬 仅测试字符串消息")
        print("5. ⏰ 仅测试时间请求")
        print("6. 🔄 持续监控模式")
        print("0. 🚪 退出程序")
        
        try:
            choice = input("\n👉 请输入选择 (0-6): ").strip()
            
            if choice == "0":
                logging.info("👋 程序退出")
                break
            elif choice == "1":
                run_full_test_suite()
            elif choice == "2":
                test_lid_closed_scenario()
            elif choice == "3":
                test_lid_opened_scenario()
            elif choice == "4":
                test_string_message_scenario()
            elif choice == "5":
                test_time_request_scenario()
            elif choice == "6":
                try:
                    interval = int(input("请输入监控间隔（秒，默认5）: ").strip() or "5")
                    monitor_lid_status_continuously(interval)
                except ValueError:
                    logging.warning("输入无效，使用默认间隔5秒")
                    monitor_lid_status_continuously(5)
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            logging.info("\n👋 用户中断，程序退出")
            break
        except Exception as e:
            logging.error(f"💥 程序执行错误: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("👋 程序被用户终止")
    except Exception as e:
        logging.critical(f"💥 程序崩溃: {e}", exc_info=True) 