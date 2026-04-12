"""本文件负责启动本地 Web 服务，并自动打开浏览器中的主界面。"""

import threading
import webbrowser

import uvicorn
from server import app


def open_main_page() -> None:
    # 使用本地 HTTP 提供页面，避免 PyInstaller onefile 下 file://.../Temp/_MEI*/ 相对路径失效
    webbrowser.open("http://127.0.0.1:8000/")


if __name__ == "__main__":
    threading.Timer(1.5, open_main_page).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
