#!/bin/bash

# 停止自动Git推送服务的脚本

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Stopping auto git push service..."

# 检查PID文件是否存在
if [ -f "auto_git_push.pid" ]; then
    PID=$(cat auto_git_push.pid)
    
    # 检查进程是否还在运行
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping process with PID: $PID"
        kill "$PID"
        
        # 等待进程结束
        sleep 2
        
        # 确认进程已结束
        if kill -0 "$PID" 2>/dev/null; then
            echo "Process still running, force killing..."
            kill -9 "$PID"
        fi
        
        echo "Auto git push service stopped."
    else
        echo "Process with PID $PID is not running."
    fi
    
    # 删除PID文件
    rm -f auto_git_push.pid
else
    echo "PID file not found."
    
    # 尝试找到并杀死相关进程
    PIDS=$(pgrep -f "auto_git_push.py")
    if [ -n "$PIDS" ]; then
        echo "Found running auto_git_push.py processes: $PIDS"
        echo "Killing them..."
        pkill -f "auto_git_push.py"
        echo "Done."
    else
        echo "No auto_git_push.py processes found."
    fi
fi

echo "Auto git push service management complete."
