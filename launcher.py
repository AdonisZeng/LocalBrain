"""
LocalBrain 启动器
当浏览器关闭时自动停止前后端服务
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import psutil


class LocalBrainLauncher:
    """
    LocalBrain 启动器
    管理前后端进程的生命周期
    """

    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.backend_dir = self.project_dir / "backend"
        self.frontend_dir = self.project_dir / "frontend"
        self.backend_port = 8000
        self.frontend_port = 5173
        self.processes = []
        self.browser_pid = None

    def log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def kill_port_process(self, port: int):
        """
        终止占用指定端口的进程
        """
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == 'LISTEN':
                try:
                    process = psutil.Process(conn.pid)
                    self.log(f"终止端口 {port} 进程: PID={process.pid}")
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    def check_port_in_use(self, port: int) -> bool:
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return True
        return False

    def start_backend(self):
        self.kill_port_process(self.backend_port)
        time.sleep(0.5)
        
        self.log("启动后端服务...")
        venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = Path(sys.executable)
        
        process = subprocess.Popen(
            [str(venv_python), "-m", "uvicorn", "app.main:app",
             "--host", "0.0.0.0", "--port", str(self.backend_port)],
            cwd=str(self.backend_dir),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        self.processes.append(process)
        return process

    def start_frontend(self):
        self.kill_port_process(self.frontend_port)
        time.sleep(0.5)
        
        self.log("启动前端服务...")
        process = subprocess.Popen(
            ["npm.cmd", "run", "dev"],
            cwd=str(self.frontend_dir),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            shell=True,
        )
        self.processes.append(process)
        return process

    def wait_for_server(self, port: int, timeout: int = 30) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_port_in_use(port):
                return True
            time.sleep(0.5)
        return False

    def find_browser_with_port(self, port: int) -> int:
        """
        查找连接到指定端口的浏览器进程 PID
        """
        browser_names = ['msedge.exe', 'chrome.exe', 'firefox.exe', 'browser.exe', 'edge.exe']
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name']
                if name and name.lower() in browser_names:
                    for conn in proc.net_connections():
                        if conn.laddr.port == port:
                            return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return None

    def is_browser_connected(self, port: int) -> bool:
        """
        检查是否有浏览器连接到指定端口
        """
        browser_names = ['msedge.exe', 'chrome.exe', 'firefox.exe', 'browser.exe', 'edge.exe']
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name']
                if name and name.lower() in browser_names:
                    for conn in proc.net_connections():
                        if conn.laddr.port == port:
                            return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return False

    def monitor_and_wait(self):
        """
        监控浏览器连接，当没有连接时停止服务
        """
        self.log("等待服务启动...")
        self.wait_for_server(self.backend_port, timeout=30)
        self.wait_for_server(self.frontend_port, timeout=30)
        
        url = f"http://localhost:{self.frontend_port}"
        self.log(f"打开浏览器: {url}")
        webbrowser.open(url)
        
        self.log("等待浏览器连接...")
        time.sleep(5)
        
        self.browser_pid = self.find_browser_with_port(self.frontend_port)
        
        if self.browser_pid:
            self.log(f"检测到浏览器进程: PID={self.browser_pid}")
            self.log("关闭浏览器窗口将自动停止所有服务")
            self.log("按 Ctrl+C 可手动停止")
            
            try:
                while True:
                    if not self.is_browser_connected(self.frontend_port):
                        time.sleep(3)
                        if not self.is_browser_connected(self.frontend_port):
                            self.log("浏览器已断开连接")
                            break
                    
                    try:
                        proc = psutil.Process(self.browser_pid)
                        if not proc.is_running():
                            self.log("浏览器已关闭")
                            break
                    except psutil.NoSuchProcess:
                        self.log("浏览器已关闭")
                        break
                    
                    time.sleep(2)
                        
            except KeyboardInterrupt:
                self.log("收到中断信号")
        else:
            self.log("未检测到浏览器进程，按 Ctrl+C 停止服务")
            self.log("提示: 如果浏览器已打开，请刷新页面以建立连接")
            
            try:
                while True:
                    if self.is_browser_connected(self.frontend_port):
                        self.browser_pid = self.find_browser_with_port(self.frontend_port)
                        if self.browser_pid:
                            self.log(f"检测到浏览器进程: PID={self.browser_pid}")
                            self.log("关闭浏览器窗口将自动停止所有服务")
                            break
                    time.sleep(2)
                
                while True:
                    if not self.is_browser_connected(self.frontend_port):
                        time.sleep(3)
                        if not self.is_browser_connected(self.frontend_port):
                            self.log("浏览器已断开连接")
                            break
                    
                    try:
                        proc = psutil.Process(self.browser_pid)
                        if not proc.is_running():
                            self.log("浏览器已关闭")
                            break
                    except psutil.NoSuchProcess:
                        self.log("浏览器已关闭")
                        break
                    
                    time.sleep(2)
                        
            except KeyboardInterrupt:
                self.log("收到中断信号")

    def cleanup(self):
        """
        清理所有进程
        """
        self.log("正在停止所有服务...")
        
        for process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                except Exception:
                    pass
        
        self.kill_port_process(self.backend_port)
        self.kill_port_process(self.frontend_port)
        
        self.log("所有服务已停止")

    def run(self):
        """
        运行启动器
        """
        self.log("=" * 50)
        self.log("LocalBrain 启动器")
        self.log("=" * 50)
        
        try:
            self.start_backend()
            self.start_frontend()
            self.monitor_and_wait()
        except Exception as e:
            self.log(f"发生错误: {e}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    launcher = LocalBrainLauncher()
    launcher.run()
