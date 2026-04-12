#!/usr/bin/env python3
"""
Quick-start script: serves the web UI on a simple HTTP server
while we get the full FastAPI backend running.
This is a standalone launcher that works even without all ML deps installed.
"""
import http.server
import socketserver
import os
import webbrowser
import threading

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Hallucination Hunter UI serving at http://localhost:{PORT}")
        threading.Timer(1.0, open_browser).start()
        httpd.serve_forever()
