"""Cross-platform launcher for the ENGINE HEALTH AI Streamlit application."""

import sys
import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent
    app_path = root_dir / "app.py"

    if not app_path.exists():
        print(f"Error: app.py not found in {root_dir}")
        sys.exit(1)

    print("=" * 60)
    print("🚀 LAUNCHING ENGINE HEALTH AI STREAMLIT APPLICATION")
    print(f"Python: {sys.executable}")
    print(f"App: {app_path}")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[ENGINE HEALTH AI] Application stopped by user.")
    except Exception as e:
        print(f"\n[ENGINE HEALTH AI] Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
