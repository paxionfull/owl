import win32pipe
import win32file
import pywintypes
import time
import logging
import datetime

def pipe_server(
    pipe_name: str = r'\\.\pipe\LidModeStatus',
    max_connections: int = 10,
    log_level: int = logging.INFO
):
    """
    命名管道服务端，按消息类型返回对应格式的响应
    
    Args:
        pipe_name: 管道名称，需与客户端一致
        max_connections: 最大客户端连接数
        log_level: 日志级别
    """
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"管道服务端启动，监听: {pipe_name}")
    
    while True:
        pipe_handle = None
        try:
            # 创建命名管道
            pipe_handle = win32pipe.CreateNamedPipe(
                pipe_name,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                max_connections,
                65536, 65536,
                0,
                None
            )
            
            logging.info("等待客户端连接...")
            win32pipe.ConnectNamedPipe(pipe_handle, None)
            logging.info("客户端已连接")
            
            # 持续处理客户端消息
            while True:
                try:
                    # 读取客户端消息
                    hr, message_bytes = win32file.ReadFile(pipe_handle, 65536)
                    
                    if not message_bytes:
                        # 客户端断开连接
                        logging.info("客户端断开连接")
                        break
                    
                    # 处理消息并生成响应
                    response = process_message(message_bytes)
                    
                    # 发送响应
                    win32file.WriteFile(pipe_handle, response)
                    logging.info(f"已发送响应: {response.hex().upper()}")
                    
                except pywintypes.error as e:
                    if e.args[0] == 109:  # ERROR_BROKEN_PIPE
                        logging.info("客户端连接已断开")
                        break
                    else:
                        logging.error(f"读取/发送消息时出错: {e}")
                        break
                        
        except pywintypes.error as e:
            logging.error(f"管道操作错误: {e}")
        except Exception as e:
            logging.error(f"未知错误: {e}", exc_info=True)
        finally:
            # 确保关闭管道句柄
            if pipe_handle:
                win32file.CloseHandle(pipe_handle)
                logging.info("管道句柄已关闭")
            
            # 短暂等待，避免CPU占用过高
            time.sleep(0.1)

def process_message(message_bytes: bytes) -> bytes:
    """
    检测消息类型并返回对应格式的响应
    - 十六进制消息：返回十六进制响应
    - 字符串消息：返回字符串响应
    
    Args:
        message_bytes: 客户端发送的原始字节数据
    
    Returns:
        响应的字节数据
    """
    try:
        # === 优先处理十六进制消息 ===
        hex_data = message_bytes.hex().upper()
        logging.info(f"收到原始字节数据: {hex_data}")
        
        # 十六进制命令处理
        if message_bytes == b'\xFD':
            return b'\xFD'  # 十六进制响应
        elif message_bytes == b'\xFE':
            return b'\xFE'  # 十六进制响应
        elif len(message_bytes) == 4:
            # 4字节十六进制命令，返回十六进制响应
            return b'\x00\x01\x02\x03'
        
        # === 尝试作为字符串处理 ===
        try:
            message_str = message_bytes.decode('utf-8')
            logging.info(f"解析为字符串消息: {message_str}")
            
            # 字符串消息处理，返回字符串响应
            if message_str.lower() in ["hello", "hi"]:
                return b"Hello from server!"  # 字符串响应
            elif message_str.lower() == "time":
                now = datetime.datetime.now()
                return now.strftime("%Y-%m-%d %H:%M:%S").encode('utf-8')  # 字符串响应
            else:
                return f"已收到字符串消息: {message_str}".encode('utf-8')  # 字符串响应
                
        except UnicodeDecodeError:
            # 字符串解码失败，确认为十六进制消息，返回十六进制格式响应
            return b'\xFF\xFF\x00\x01'  # 示例十六进制响应
        
    except Exception as e:
        logging.error(f"处理消息时出错: {e}")
        return b"ERROR: " + str(e).encode('utf-8')  # 错误响应为字符串

if __name__ == "__main__":
    try:
        pipe_server()
    except KeyboardInterrupt:
        logging.info("服务器被用户手动终止")
    except Exception as e:
        logging.critical(f"服务器崩溃: {e}", exc_info=True)


