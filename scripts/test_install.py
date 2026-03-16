import compileall
from pathlib import Path


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    required_paths = [
        root_dir / "backend" / "main.py",
        root_dir / "frontend" / "package.json",
        root_dir / "requirements.txt",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Install test failed. Missing files: {missing}")

    compiled = compileall.compile_dir(root_dir / "backend", quiet=1)
    if not compiled:
        raise SystemExit("Install test failed. Backend compilation did not succeed.")

    print("Install test passed: required files exist and backend compiles.")
