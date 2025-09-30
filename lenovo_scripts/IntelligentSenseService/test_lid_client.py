import win32file
import win32pipe
import pywintypes
import time
import logging
import os

def create_test_file():
    """创建测试文件hello_lenovo.txt"""
    try:
        # 在当前目录创建空文件
        with open("hello_lenovo.txt", "w") as f:
            f.write("")  # 创建空文件
        print("✅ 成功创建文件: hello_lenovo.txt")
        return True
    except Exception as e:
        print(f"❌ 创建文件失败: {e}")
        return False

def test_lid_status_client():
    """
    测试客户端 - 模拟检测笔记本盖子状态
    """
    pipe_name = r'\\.\pipe\LidModeStatus'
    
    print("🔍 正在连接到联想智能感知服务...")
    
    try:
        # 连接到命名管道
        pipe_handle = win32file.CreateFile(
            pipe_name,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
        )
        
        print("✅ 成功连接到服务器")
        
        # 测试场景1: 发送盖子合上命令 (假设 \xFD 代表盖子合上)
        print("\n🔒 测试场景: 笔记本盖子合上")
        lid_closed_command = b'\xFD'
        
        # 发送命令
        win32file.WriteFile(pipe_handle, lid_closed_command)  # type: ignore
        print(f"📤 发送命令: {lid_closed_command.hex().upper()}")
        
        # 读取响应
        hr, response = win32file.ReadFile(pipe_handle, 65536)  # type: ignore
        if isinstance(response, bytes):
            print(f"📥 收到响应: {response.hex().upper()}")
            
            # 检查响应是否表示盖子合上状态
            if response == b'\xFD':
                print("🎯 检测到盖子合上状态!")
                create_test_file()
            else:
                print(f"⚠️  未识别的响应: {response.hex().upper()}")
        else:
            print(f"📥 收到响应: {response}")
        
        # 测试场景2: 发送盖子打开命令 (假设 \xFE 代表盖子打开)
        print("\n🔓 测试场景: 笔记本盖子打开") 
        lid_open_command = b'\xFE'
        
        win32file.WriteFile(pipe_handle, lid_open_command)
        print(f"📤 发送命令: {lid_open_command.hex().upper()}")
        
        hr, response = win32file.ReadFile(pipe_handle, 65536)
        if isinstance(response, bytes):
            print(f"📥 收到响应: {response.hex().upper()}")
            
            if response == b'\xFE':
                print("🎯 检测到盖子打开状态!")
                # 可以在这里添加清理逻辑，比如删除文件
                if os.path.exists("hello_lenovo.txt"):
                    os.remove("hello_lenovo.txt")
                    print("🗑️  已删除测试文件")
        else:
            print(f"📥 收到响应: {response}")
        
        # 测试场景3: 发送字符串消息
        print("\n💬 测试场景: 发送字符串消息")
        hello_message = "hello".encode('utf-8')
        
        win32file.WriteFile(pipe_handle, hello_message)
        print(f"📤 发送消息: {hello_message.decode('utf-8')}")
        
        hr, response = win32file.ReadFile(pipe_handle, 65536)
        if isinstance(response, bytes):
            try:
                print(f"📥 收到响应: {response.decode('utf-8')}")
            except UnicodeDecodeError:
                print(f"📥 收到十六进制响应: {response.hex().upper()}")
        else:
            print(f"📥 收到响应: {response}")
        
    except pywintypes.error as e:
        if e.args[0] == 2:  # ERROR_FILE_NOT_FOUND
            print("❌ 无法连接到服务器，请确保pipeServer.py正在运行")
        else:
            print(f"❌ 管道连接错误: {e}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        try:
            win32file.CloseHandle(pipe_handle)
            print("🔐 连接已关闭")
        except:
            pass

def monitor_lid_status():
    """
    持续监控模式 - 每隔5秒检查一次盖子状态
    """
    print("🔄 启动持续监控模式 (Ctrl+C 停止)")
    
    try:
        while True:
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - 检查盖子状态...")
            test_lid_status_client()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n⏹️  监控已停止")

if __name__ == "__main__":
    print("=== 联想笔记本盖子状态测试客户端 ===")
    print("1. 单次测试")
    print("2. 持续监控")
    
    choice = input("请选择模式 (1/2): ").strip()
    
    if choice == "2":
        monitor_lid_status()
    else:
        test_lid_status_client() 