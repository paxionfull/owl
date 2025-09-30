import win32con
import win32gui
import win32api
import ctypes
from ctypes import wintypes
import struct
import webbrowser
import threading
import subprocess  
import win32file
import win32pipe
import pywintypes
import time
import logging
import os
from pathlib import Path
import sys

logging.basicConfig(level=logging.DEBUG)
# GUID for lid switch state change
GUID_LIDSWITCH_STATE_CHANGE = '{BA3E0F4D-B817-4094-A2D1-D56379E6A0F3}'

# Constants
WM_POWERBROADCAST = 0x0218
PBT_POWERSETTINGCHANGE = 0x8013

# Define POWERBROADCAST_SETTING structure
class POWERBROADCAST_SETTING(ctypes.Structure):
    _fields_ = [
        ('PowerSetting', ctypes.c_ubyte * 16),
        ('DataLength', wintypes.DWORD),
        ('Data', wintypes.BYTE * 1)  # placeholder, we will handle manually
    ]

# Convert GUID string to bytes
import uuid
def guid_bytes(guid_str):
    guid = uuid.UUID(guid_str)
    return guid.bytes_le  # little-endian byte order


def execute_python_script(script_path: str, working_dir: str | None = None, log_file: str | None = None) -> bool:
    """
    执行Python脚本并保存日志
    
    Args:
        script_path: Python脚本的路径
        working_dir: 工作目录，如果不指定则使用脚本所在目录
        log_file: 日志文件路径，如果不指定则使用默认路径
    
    Returns:
        执行是否成功
    """
    try:
        # 如果没有指定工作目录，使用脚本所在目录的父目录
        if working_dir is None:
            script_dir = Path(script_path).parent.resolve()
            working_dir = str(script_dir.parent.resolve())  # 上级目录，因为脚本在子目录中
        
        # 构建完整的脚本路径
        full_script_path = Path(working_dir) / script_path
        
        if not full_script_path.exists():
            logging.error(f"Python脚本不存在: {full_script_path}")
            return False
        
        # 设置日志文件路径
        if log_file is None:
            script_name = Path(script_path).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/{script_name}_{timestamp}.log"
        
        # 确保日志目录存在
        log_dir = Path(log_file).parent
        log_dir.mkdir(exist_ok=True)
        
        # 构建执行命令
        python_exe = sys.executable  # 使用当前Python解释器
        cmd = [python_exe, str(full_script_path)]
        
        logging.info(f"准备执行Python脚本: {' '.join(cmd)}")
        logging.info(f"工作目录: {working_dir}")
        logging.info(f"日志文件: {log_file}")
        
        # 在后台执行脚本，捕获输出
        process = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
            text=True,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # 在新控制台窗口中运行
        )
        
        logging.info(f"Python脚本已启动，进程ID: {process.pid}")
        
        # 启动后台线程来处理日志输出
        def log_output():
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"脚本执行开始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"执行命令: {' '.join(cmd)}\n")
                    f.write(f"工作目录: {working_dir}\n")
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
                                f.write(output)
                                f.flush()
                                # 同时写入主日志
                                logging.info(f"[{script_path}] {output.strip()}")
                    else:
                        f.write("无法捕获脚本输出\n")
                        logging.warning("无法捕获脚本输出")
                    
                    # 获取返回码
                    return_code = process.poll()
                    f.write("=" * 50 + "\n")
                    f.write(f"脚本执行结束: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"返回码: {return_code}\n")
                    
                    if return_code == 0:
                        logging.info(f"Python脚本执行成功，日志已保存到: {log_file}")
                    else:
                        logging.error(f"Python脚本执行失败，返回码: {return_code}，日志已保存到: {log_file}")
                        
            except Exception as e:
                logging.error(f"写入日志文件时发生错误: {e}")
        
        # 启动日志处理线程
        log_thread = threading.Thread(target=log_output, daemon=True)
        log_thread.start()
        
        return True
        
    except Exception as e:
        logging.error(f"执行Python脚本时发生错误: {e}")
        return False


def set_power_timeouts():
    """Set power timeouts to 1 minute for both AC and DC power"""
    logging.info("set_power_timeouts.start")
    subprocess.run(["powercfg", "/change", "standby-timeout-ac", "1"], check=True)
    subprocess.run(["powercfg", "/change", "standby-timeout-dc", "1"], check=True)
    subprocess.run(["powercfg", "/change", "monitor-timeout-ac", "1"], check=True)
    subprocess.run(["powercfg", "/change", "monitor-timeout-dc", "1"], check=True)
    time.sleep(60)

def check_power_requests():
    """Check for active power requests that prevent Modern Standby"""
    result = subprocess.run(["powercfg", "/requests"], capture_output=True, text=True)
    blockers = ["DISPLAY", "SYSTEM", "AWAYMODE", "EXECUTION"]
    if any(keyword in result.stdout for keyword in blockers):
        logging.info("Cannot enter Modern Standby due to active requests:")
        logging.info(result.stdout)
        return False
    return True

def turn_off_monitor():
    """Turn off the monitor using Windows API"""
    HWND_BROADCAST = 0xFFFF
    WM_SYSCOMMAND = 0x0112
    SC_MONITORPOWER = 0xF170
    
    # Define SendMessage function
    SendMessage = ctypes.windll.user32.SendMessageW
    SendMessage.argtypes = (
        ctypes.wintypes.HWND,    # hWnd
        ctypes.wintypes.UINT,    # Msg
        ctypes.wintypes.WPARAM,  # wParam
        ctypes.wintypes.LPARAM   # lParam
    )
    SendMessage.restype = ctypes.c_long  
    
    # Send monitor power off command
    result = SendMessage(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)
    logging.info(f"Screen off command sent (return value: {result})")



def enter_modern_standby():
    """Main function to prepare and enter Modern Standby"""
    try:
        # 1. Set power timeouts and wait
        set_power_timeouts()
        logging.info("set_power_timeouts.")
        # 2. Check for blocking processes
        if not check_power_requests():
            logging.info("check_power_requests---false.")
            return False
        logging.info("check_power_requests---true.")
        # 3. Turn off monitor
        turn_off_monitor()
        logging.info("turn_off_monitor---true.")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Power configuration failed: {e}")
        return False
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False
    
def delete_file(file_path: str) -> bool:
    """
    Args:
        file_path: 要删除的文件路径
    """
    # 规范化路径
    path = Path(file_path).resolve()
    
    # 安全检查
    if not path.exists():
        logging.warning(f"文件不存在: {path}")
        return False
    
    if path.is_dir():
        logging.error(f"路径指向目录而非文件: {path}")
        return False
    
    try:
        # 尝试删除文件
        os.remove(path)
        logging.info(f"文件已成功删除: {path}")
        return True
        
    except PermissionError:
        logging.error(f"权限不足，无法删除文件: {path}")
    except FileNotFoundError:
        logging.warning(f"文件在删除前已被移除: {path}")
    except Exception as e:
        logging.error(f"删除文件时发生错误: {e}")
    
    return False
def open_html_file_in_browser(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 '{file_path}' 不存在")
        return
    
    # 将文件路径转换为绝对路径，并确保使用文件协议
    absolute_path = os.path.abspath(file_path)
    url = f"file://{absolute_path}"
    
    # 使用默认浏览器打开
    webbrowser.open(url)
    time.sleep(5)


def send_message_to_pipe(
    pipe_name: str = r'\\.\pipe\LidModeStatus',
    message: int | str | bytes = "Hello from client!",  # 默认按字符串处理
    max_attempts: int = 5,
    retry_interval: int = 1,
    timeout: int = 10
) -> bytes:
    """
    向命名管道发送消息并接收响应
    - bytes类型：直接作为十六进制数据发送
    - str类型：按普通字符串（UTF-8编码）发送
    - int类型：转换为单字节十六进制数据
    
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
        
        # 连接到管道（保持原有逻辑不变）
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
                        logging.info(f"管道未找到，{retry_interval}秒后重试...")
                        time.sleep(retry_interval)
                    else:
                        raise
                else:
                    raise
        
        # 设置管道为消息模式（保持原有逻辑不变）
        win32pipe.SetNamedPipeHandleState(
            pipe,
            win32pipe.PIPE_READMODE_MESSAGE,
            None,
            None
        )
        
        # 发送消息（保持原有逻辑不变）
        win32file.WriteFile(pipe, message_bytes)
        logging.info(f"已发送消息（{len(message_bytes)}字节）")
        
        # 接收响应（保持原有逻辑不变）
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                hr, data = win32file.ReadFile(pipe, 65536)
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
            win32file.CloseHandle(pipe)
    
    return response
def check_file_repeatedly(file_path, interval=10):
    """
    每隔 interval 秒检查一次文件是否存在
    """
    if os.path.exists(file_path):
        logging.info(f"[INFO] 文件已生成：{file_path}")
        #  机器进入睡眠
        enter_modern_standby()
    else:
        logging.info(f"[INFO] 文件未找到，{interval} 秒后重试...")
        threading.Timer(interval, check_file_repeatedly, args=(file_path, interval)).start()

# Register notification
def register_power_notification(hwnd):
    user32 = ctypes.windll.user32
    powrprof = ctypes.windll.PowrProf

    RegisterPowerSettingNotification = user32.RegisterPowerSettingNotification
    RegisterPowerSettingNotification.restype = wintypes.HANDLE
    RegisterPowerSettingNotification.argtypes = [wintypes.HWND, ctypes.POINTER(ctypes.c_byte), wintypes.DWORD]

    guid = guid_bytes(GUID_LIDSWITCH_STATE_CHANGE)
    return RegisterPowerSettingNotification(hwnd, ctypes.cast(ctypes.create_string_buffer(guid), ctypes.POINTER(ctypes.c_byte)), 0)

# Message loop and handler
def wnd_proc(hwnd, msg, wparam, lparam):
    if msg == WM_POWERBROADCAST and wparam == PBT_POWERSETTINGCHANGE:
        data_ptr = ctypes.cast(lparam, ctypes.POINTER(POWERBROADCAST_SETTING))
        power_setting = bytes(data_ptr.contents.PowerSetting[:16])
        lid_state = data_ptr.contents.Data[0]
        if power_setting == guid_bytes(GUID_LIDSWITCH_STATE_CHANGE):
            state_text = "Opened" if lid_state else "Closed"
            logging.info(f"Lid state changed: {state_text} (lid_state={lid_state})")
            task_list = r'D:\Test\justtest\dailyschedule.html'
            if lid_state:#open
                logging.info("🔓 执行盖子打开逻辑...")
                response = send_message_to_pipe(
                                message=0xFD  # 转换为单字节0XEF
                                )
                logging.info(f"单字节响应: {response.hex().upper()}")
                #打开task list 完成
                if os.path.exists(task_list):
                    open_html_file_in_browser(task_list)
            else:#close
                logging.info("🔒 执行盖子合上逻辑...")
                response = send_message_to_pipe(
                                message=0xFE  # 转换为单字节0XEF
                                )
                logging.info(f"单字节响应: {response.hex().upper()}")
                
                # 执行 run_lenovo_workforce.py 脚本
                logging.info("检测到合盖动作，开始执行Python脚本...")
                # 为脚本指定日志文件，包含时间戳以便区分不同的执行
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file_path = rf"D:\workspace\projects\owl\logs\run_lenovo_workforce_{timestamp}.log"
                script_success = execute_python_script(
                    script_path="run_lenovo_workforce.py", 
                    working_dir=r"D:\workspace\projects\owl",
                    log_file=log_file_path
                )
                # script_success = True
                # logging.info("执行python脚本完毕")
                
                if script_success:
                    logging.info("Python脚本执行成功")
                else:
                    logging.error("Python脚本执行失败")
                
                #删除文件完成   检测文件  检测到文件   进入睡眠
                delete_file(task_list)
                check_file_repeatedly(task_list)
    return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

def main():
    wc = win32gui.WNDCLASS()
    hinst = win32api.GetModuleHandle()
    wc.lpfnWndProc = wnd_proc
    wc.lpszClassName = "PowerNotifyWindow"
    class_atom = win32gui.RegisterClass(wc)
    hwnd = win32gui.CreateWindow(wc.lpszClassName, "Power Notification", 0, 0, 0, 0, 0, 0, 0, hinst, None)
    register_power_notification(hwnd)
    logging.info("Listening for lid switch events. Press Ctrl+C to exit.")
    win32gui.PumpMessages()

if __name__ == "__main__":
    main()
