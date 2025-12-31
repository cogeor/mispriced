import http.server
import socketserver
import os
import webbrowser
import sys

PORT = 8000
DIRECTORY = "web"

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def serve():
    # web folder is at ../web relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.join(os.path.dirname(script_dir), 'web')
    
    print(f"Serving Dashboard from: {web_dir}")
    print(f"URL: http://localhost:{PORT}")
    
    # Change to parent dir so we can serve 'web' as root or serve from inside web
    # User wants to access dashboard. Serving INSIDE web/ public folder?
    # No, web/ contains index.html.
    
    os.chdir(os.path.dirname(web_dir))
    
    # Actually, let's serve 'web' as the root.
    # But my script sets directory=DIRECTORY.
    
    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            print("Dashboard running. Press Ctrl+C to stop.")
            webbrowser.open(f"http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    except OSError as e:
        print(f"Error: {e}. Maybe port {PORT} is in use?")

if __name__ == "__main__":
    serve()
