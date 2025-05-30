#!/usr/bin/env python3
"""
Automated Git Push Script
Automatically commits and pushes changes every 1-2 hours (random interval)
"""

import subprocess
import time
import os
import logging
import random
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_git_push.log'),
        logging.StreamHandler()
    ]
)

def run_git_command(command):
    """运行git命令并返回结果"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        logging.error(f"Error running command '{command}': {e}")
        return False, "", str(e)

def git_auto_push():
    """自动git推送函数"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logging.info(f"Starting auto git push at {timestamp}")
    
    # 检查是否有更改
    success, stdout, stderr = run_git_command("git status --porcelain")
    if not success:
        logging.error(f"Failed to check git status: {stderr}")
        return False
    
    if not stdout.strip():
        logging.info("No changes to commit")
        return True
    
    logging.info(f"Found changes:\n{stdout}")
    
    # 添加所有更改
    success, stdout, stderr = run_git_command("git add -A")
    if not success:
        logging.error(f"Failed to add changes: {stderr}")
        return False
    
    logging.info("Added all changes")
    
    # 提交更改
    commit_message = f"Auto commit - {timestamp}"
    success, stdout, stderr = run_git_command(f'git commit -m "{commit_message}"')
    if not success:
        logging.error(f"Failed to commit changes: {stderr}")
        return False
    
    logging.info(f"Committed changes with message: {commit_message}")
    
    # 推送到远程仓库
    success, stdout, stderr = run_git_command("git push")
    if not success:
        logging.error(f"Failed to push changes: {stderr}")
        return False
    
    logging.info("Successfully pushed changes to remote repository")
    return True

def main():
    """主函数 - 每1-2小时随机间隔运行一次git推送"""
    logging.info("Starting auto git push daemon")
    logging.info("Will check for changes and push every 1-2 hours (random interval)")
    logging.info("Press Ctrl+C to stop")
    
    # 首次运行
    git_auto_push()
    
    try:
        while True:
            # 随机等待1-2小时 (3600-7200秒)
            wait_time = random.randint(3600, 7200)  # 1小时到2小时
            wait_hours = wait_time / 3600
            logging.info(f"Waiting {wait_hours:.2f} hours until next push...")
            
            time.sleep(wait_time)
            git_auto_push()
            
    except KeyboardInterrupt:
        logging.info("Auto git push daemon stopped by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
