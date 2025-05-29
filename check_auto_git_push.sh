#!/bin/bash

# 自动Git推送服务状态检查脚本

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "=== Auto Git Push Service Status ==="
echo "Timestamp: $(date)"
echo ""

# 检查PID文件
if [ -f "auto_git_push.pid" ]; then
    PID=$(cat auto_git_push.pid)
    echo "PID file found: $PID"
    
    # 检查进程是否运行
    if kill -0 "$PID" 2>/dev/null; then
        echo "✅ Service is RUNNING (PID: $PID)"
        
        # 显示进程信息
        echo ""
        echo "Process details:"
        ps -p "$PID" -o pid,ppid,cmd,etime,pcpu,pmem
        
    else
        echo "❌ Service is NOT RUNNING (stale PID file)"
    fi
else
    echo "❌ PID file not found"
    
    # 检查是否有相关进程在运行
    PIDS=$(pgrep -f "auto_git_push.py")
    if [ -n "$PIDS" ]; then
        echo "⚠️  Found orphaned processes: $PIDS"
    fi
fi

echo ""
echo "=== Recent Log Activity ==="
if [ -f "auto_git_push.log" ]; then
    echo "Last 5 log entries:"
    tail -5 auto_git_push.log
else
    echo "No log file found"
fi

echo ""
echo "=== Next Auto Push ==="
if [ -f "auto_git_push.pid" ] && kill -0 "$(cat auto_git_push.pid)" 2>/dev/null; then
    # 计算下次推送时间（假设每30分钟）
    if [ -f "auto_git_push.log" ]; then
        LAST_PUSH=$(grep "Successfully pushed" auto_git_push.log | tail -1 | cut -d' ' -f1-2)
        if [ -n "$LAST_PUSH" ]; then
            echo "Last successful push: $LAST_PUSH"
            # 这里可以添加更复杂的时间计算，但bash日期计算比较复杂
            echo "Next push: ~30 minutes after last push"
        fi
    fi
else
    echo "Service not running - no scheduled pushes"
fi

echo ""
echo "=== Commands ==="
echo "Start service:  ./start_auto_git_push.sh"
echo "Stop service:   ./stop_auto_git_push.sh"
echo "View logs:      tail -f auto_git_push.log"
echo "Check status:   ./check_auto_git_push.sh"
