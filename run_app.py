import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

def main():
    ROOT = Path(__file__).resolve().parent
    SUB_DIR = ROOT / "AI-Research-Collaborator-main"

    if SUB_DIR.exists() and (SUB_DIR / "server.py").exists():
        target_dir = SUB_DIR
    else:
        target_dir = ROOT

    print(f"Launching AI Research Collaborator Server from {target_dir} ...")
    
    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))
    os.chdir(target_dir)

    try:
        import uvicorn
    except ImportError:
        import subprocess
        print("Installing uvicorn and fastapi...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
        import uvicorn

    def open_browser():
        time.sleep(1.5)
        print("Opening browser at http://localhost:8000 ...")
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    # Windows par reload=False se multiprocessing crash nahi hota
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(target_dir))

if __name__ == "__main__":
    main()
