# 联想智能感知服务 (IntelligentSenseService)

## 概述
这是一个Windows命名管道服务系统，用于检测和响应联想笔记本的盖子状态。

## 文件说明

### `pipeServer.py` - 管道服务器
- **功能**: 创建命名管道服务器，监听盖子状态查询
- **管道名称**: `\\.\pipe\LidModeStatus`
- **支持命令**:
  - `\xFD` - 盖子合上状态查询
  - `\xFE` - 盖子打开状态查询
  - 字符串消息 (如 "hello", "time")

### `test_lid_client.py` - 测试客户端
- **功能**: 连接到服务器并测试盖子状态检测
- **特性**: 当检测到盖子合上时，自动创建 `hello_lenovo.txt` 文件

## 使用方法

### 1. 启动服务器
```bash
cd lenovo_scripts/IntelligentSenseService
python pipeServer.py
```

服务器将显示：
```
INFO - 管道服务端启动，监听: \\.\pipe\LidModeStatus
INFO - 等待客户端连接...
```

### 2. 运行测试客户端
在另一个终端中运行：
```bash
python test_lid_client.py
```

选择测试模式：
- **1**: 单次测试 - 执行一次完整的测试流程
- **2**: 持续监控 - 每5秒检查一次盖子状态

### 3. 测试场景

#### 场景1: 盖子合上检测
- 发送命令: `\xFD`
- 期望响应: `\xFD`
- 动作: 创建 `hello_lenovo.txt` 文件

#### 场景2: 盖子打开检测
- 发送命令: `\xFE`
- 期望响应: `\xFE`
- 动作: 删除 `hello_lenovo.txt` 文件

#### 场景3: 字符串消息测试
- 发送消息: "hello"
- 期望响应: "Hello from server!"

## 示例输出

### 服务器端
```
2024-01-XX XX:XX:XX - INFO - 管道服务端启动，监听: \\.\pipe\LidModeStatus
2024-01-XX XX:XX:XX - INFO - 等待客户端连接...
2024-01-XX XX:XX:XX - INFO - 客户端已连接
2024-01-XX XX:XX:XX - INFO - 收到原始字节数据: FD
2024-01-XX XX:XX:XX - INFO - 已发送响应: FD
```

### 客户端
```
=== 联想笔记本盖子状态测试客户端 ===
🔍 正在连接到联想智能感知服务...
✅ 成功连接到服务器

🔒 测试场景: 笔记本盖子合上
📤 发送命令: FD
📥 收到响应: FD
🎯 检测到盖子合上状态!
✅ 成功创建文件: hello_lenovo.txt
```

## 依赖项
- `pywin32` - Windows API 支持
- Python 3.6+

## 安装依赖
```bash
pip install pywin32
```

## 注意事项
1. 此服务仅在Windows系统上运行
2. 需要管理员权限来创建命名管道
3. 确保防火墙不会阻止管道通信
4. 服务器必须先启动，客户端才能连接成功 