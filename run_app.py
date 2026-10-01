import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent

    target_dir = root
    alt_dir = root / "AI-Research-Collaborator-main"
    if alt_dir.exists() and (alt_dir / "server.py").exists():
        target_dir = alt_dir

    print(f"Launching AI Research Collaborator from: {target_dir}")
    if str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))
    os.chdir(str(target_dir))

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print("Installing required dependencies from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt")], check=False)
        try:
            import uvicorn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("FastAPI/Uvicorn installation failed") from exc

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    def open_browser() -> None:
        time.sleep(1.5)
        if os.getenv("BROWSER_OPEN", "1") == "1":
            print(f"Opening browser at http://{host}:{port} ...")
            webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("server:app", host=host, port=port, reload=False, app_dir=str(target_dir))


if __name__ == "__main__":
    main()
