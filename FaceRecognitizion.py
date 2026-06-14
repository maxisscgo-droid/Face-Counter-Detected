import webview
import subprocess
import sys
import os

class Api:
    def open_camera(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.py')
        subprocess.Popen([sys.executable, script])

api = Api()
webview.create_window("Mi App", "acced.html", js_api=api)
webview.start()
