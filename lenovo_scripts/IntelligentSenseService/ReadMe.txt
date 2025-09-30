
运行IntelligentSenseService.py
1.检测到屏幕合盖动作，发送合盖消息0xFE给无穹，开始检测'D:\Test\justtest\dailyschedule.html'是否已生成，
   dailyschedule.html已经生成，笔记本进入睡眠状态
   未生成，继续每过10秒，检测一次
2.检测到到开盖信息，发送开盖消息0xFD 给无穹，并且使用默认浏览器打开D:\Test\justtest\dailyschedule.html

pipeServer.py是模拟的接收管道消息，仅供参考接收管道消息用。