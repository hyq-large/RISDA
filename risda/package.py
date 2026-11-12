import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from .environment import ROOT


def source_files(root):
    top = ("README.md", "pyproject.toml", "run_single.sh", "run_train.sh", ".gitignore")
    files = [root / name for name in top]
    suffixes = {".py", ".md", ".yaml", ".example"}
    for directory in ("risda", "appa", "configs", "docs"):
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts
        )
    files = sorted(set(files))
    patterns = (
        r"/(?:inspire|home|root)/",
        r"tenant_[a-zA-Z0-9-]+",
        r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+",
    )
    for path in files:
        if path.is_symlink() or root.resolve() not in path.resolve().parents:
            raise ValueError(f"Refusing external or symlinked source: {path}")
        content = path.read_text()
        if any(re.search(pattern, content) for pattern in patterns):
            raise ValueError(f"Potential identifying text in {path.relative_to(root)}")
    return files


def build_archive(root, output):
    files = source_files(root)
    manifest = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = str(path.relative_to(root))
            data = path.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo("RISDA/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
        archive.writestr("RISDA/SHA256SUMS.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build an anonymous source ZIP without local artifacts or Git history"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "dist/risda-anonymous.zip")
    args = parser.parse_args()
    manifest = build_archive(ROOT, args.output)
    print(f"Packaged {len(manifest)} files: {args.output}")


if __name__ == "__main__":
    main()
