"""Verify deterministic train/dev benchmark artifacts."""
from __future__ import annotations
import hashlib, subprocess, sys, tempfile
from pathlib import Path
def digest_tree(root: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.iterdir()) if path.is_file()}
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary); first, second = root / "first", root / "second"
    command = [sys.executable, "-m", "ctrag.benchmarks", "--seeds", "7", "42", "2024", "--ks", "1", "3", "5", "10", "--output"]
    subprocess.run([*command, str(first)], check=True); subprocess.run([*command, str(second)], check=True)
    if digest_tree(first) != digest_tree(second): raise SystemExit("reproduction mismatch")
    print("reproduction verified")
