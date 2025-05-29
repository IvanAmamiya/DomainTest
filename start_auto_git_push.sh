#!/bin/bash

# 自动Git推送启动脚本
# 在后台运行git自动推送服务

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Starting auto git push service..."
echo "Script directory: $SCRIPT_DIR"
echo "Logs will be written to: auto_git_push.log"
echo ""

# 检查Python脚本是否存在
if [ ! -f "auto_git_push.py" ]; then
    echo "Error: auto_git_push.py not found!"
    exit 1
fi

# 使脚本可执行
chmod +x auto_git_push.py

# 在后台运行Python脚本
nohup python3 auto_git_push.py > auto_git_push_output.log 2>&1 &

# 获取进程ID
PID=$!
echo "Auto git push service started with PID: $PID"
echo "$PID" > auto_git_push.pid

echo ""
echo "To stop the service, run: kill $PID"
echo "Or use: ./stop_auto_git_push.sh"
echo ""
echo "To monitor the logs in real-time:"
echo "  tail -f auto_git_push.log"
echo "  tail -f auto_git_push_output.log"
