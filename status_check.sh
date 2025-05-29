#!/bin/bash

echo "=== Auto Git Push Service Status ==="
echo "Timestamp: $(date)"
echo ""

# 检查PID文件
if [ -f "auto_git_push.pid" ]; then
    PID=$(cat auto_git_push.pid)
    echo "PID file found: $PID"
    
    # 检查进程是否运行
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Service is RUNNING (PID: $PID)"
        echo ""
        echo "Process details:"
        ps -p "$PID" -o pid,ppid,cmd,etime,pcpu,pmem
    else
        echo "❌ Service is NOT RUNNING (stale PID file)"
    fi
else
    echo "❌ PID file not found"
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
echo "=== Commands ==="
echo "Start service:  ./start_auto_git_push.sh"
echo "Stop service:   ./stop_auto_git_push.sh"
echo "View logs:      tail -f auto_git_push.log"
