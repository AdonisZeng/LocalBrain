"""
LocalBrain Build Script

Builds the LocalBrain application into a standalone .exe file.

Usage:
    python build.py           # Build in current directory
    python build.py clean     # Clean build artifacts
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

# Project root is the directory containing this script
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SPEC_FILE = PROJECT_ROOT / "LocalBrain.spec"


def get_python():
    """Get the Python executable to use."""
    if VENV_PYTHON.exists():
        return VENV_PYTHON
    return Path(sys.executable)


def run_command(cmd, cwd=None, check=True, capture_output=False):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(str(c) if isinstance(c, Path) else c for c in cmd)}")
    if cwd:
        print(f"CWD: {cwd}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=False,
        capture_output=capture_output,
        text=True
    )
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        sys.exit(1)
    return result


def build_frontend():
    """Build the frontend React application."""
    print("\n" + "="*60)
    print("Building frontend...")
    print("="*60)

    # Check if node_modules exists
    if not (FRONTEND_DIR / "node_modules").exists():
        print("Installing frontend dependencies...")
        run_command(["npm.cmd", "install"], cwd=FRONTEND_DIR)

    # Build frontend
    print("Building frontend with Vite...")
    run_command(["npm.cmd", "run", "build"], cwd=FRONTEND_DIR)

    dist_dir = FRONTEND_DIR / "dist"
    if not dist_dir.exists():
        print(f"ERROR: Frontend build failed - {dist_dir} not found")
        sys.exit(1)

    print(f"Frontend built successfully: {dist_dir}")
    return dist_dir


def install_pyinstaller():
    """Install PyInstaller if not already installed."""
    print("\n" + "="*60)
    print("Checking PyInstaller...")
    print("="*60)

    python = get_python()

    # Check if pyinstaller is available
    result = subprocess.run(
        [str(python), "-m", "PyInstaller", "--version"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Installing PyInstaller via uv...")
        # Use uv to install into the project venv
        uv_exe = Path("D:/Software/uv/uv.exe")

        venv_path = PROJECT_ROOT / ".venv"
        subprocess.run([str(uv_exe), "pip", "install", "pyinstaller", "--python", str(venv_path)], check=True)
    else:
        print(f"PyInstaller already installed: {result.stdout.strip()}")


def build_exe():
    """Build the .exe using PyInstaller."""
    print("\n" + "="*60)
    print("Building .exe with PyInstaller...")
    print("="*60)

    python = get_python()

    # Run PyInstaller with the spec file
    cmd = [
        str(python),
        "-m", "PyInstaller",
        str(SPEC_FILE),
        "--workpath", str(PROJECT_ROOT / "build"),
        "--distpath", str(PROJECT_ROOT / "dist"),
        "-y",  # Remove output directory without confirmation
    ]

    run_command(cmd)

    # Check for exe (output is in dist/LocalBrain/)
    exe_path = PROJECT_ROOT / "dist" / "LocalBrain" / "LocalBrain.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"Build successful!")
        print(f"Executable: {exe_path}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"{'='*60}")
    else:
        print(f"ERROR: Build failed - exe not found at {exe_path}")
        sys.exit(1)


def clean():
    """Clean build artifacts."""
    print("\n" + "="*60)
    print("Cleaning build artifacts...")
    print("="*60)

    dirs_to_remove = [
        PROJECT_ROOT / "build",
        PROJECT_ROOT / "dist",
        PROJECT_ROOT / "__pycache__",
        BACKEND_DIR / "app" / "__pycache__",
    ]

    # Also remove any LocalBrain-xxx directories
    for item in PROJECT_ROOT.iterdir():
        if item.name.startswith("LocalBrain-") and item.is_dir():
            dirs_to_remove.append(item)

    for d in dirs_to_remove:
        if d.exists():
            print(f"Removing {d}")
            shutil.rmtree(d, ignore_errors=True)

    # Remove spec file (we'll regenerate it if needed)
    # Actually keep the spec file, it's part of the project

    # Remove .pyc files
    for pyc_file in PROJECT_ROOT.rglob("*.pyc"):
        print(f"Removing {pyc_file}")
        pyc_file.unlink()

    # Remove __pycache__ directories
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        print(f"Removing {pycache}")
        shutil.rmtree(pycache, ignore_errors=True)

    print("Clean complete!")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
        return

    print("="*60)
    print("LocalBrain Build Script")
    print("="*60)

    # Step 1: Build frontend
    frontend_dist = build_frontend()

    # Step 2: Install PyInstaller
    install_pyinstaller()

    # Step 3: Build .exe
    build_exe()

    print("\n" + "="*60)
    print("All done!")
    print("="*60)
    print("\nTo run the application:")
    print(f"  {PROJECT_ROOT / 'dist' / 'LocalBrain' / 'LocalBrain.exe'}")
    print("\nData and logs will be created alongside the .exe")


if __name__ == "__main__":
    main()
