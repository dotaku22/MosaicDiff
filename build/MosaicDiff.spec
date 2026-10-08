# -*- mode: python ; coding: utf-8 -*-
"""Freeze MosaicDiff. PyInstaller executes this file from the build folder."""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

root = os.path.abspath(os.path.join(SPECPATH, ".."))
if root not in sys.path:
    sys.path.insert(0, root)

datas = [
    (os.path.join(root, "mosaicdiff", "h3_worker.py"), "mosaicdiff"),
    (os.path.join(root, "models", "README.txt"), "models"),
]
license_dir = os.path.join(root, "licenses")
for name in os.listdir(license_dir):
    datas.append((os.path.join(license_dir, name), "licenses"))
nodes = os.path.join(root, "comfy_nodes")
for dirpath, _, filenames in os.walk(nodes):
    for name in filenames:
        if name.endswith(".pyc"):
            continue
        rel = os.path.relpath(dirpath, nodes)
        dest = "comfy_nodes" if rel == "." else os.path.join("comfy_nodes", rel)
        datas.append((os.path.join(dirpath, name), dest))
binaries = []
hidden = collect_submodules("mosaicdiff")
# The pip package is onnxruntime-gpu. The import name stays onnxruntime.
hidden += ["mmengine", "onnxruntime", "customtkinter", "av", "cv2"]

for package in ("torch", "customtkinter", "av", "onnxruntime", "mmengine", "huggingface_hub", "cv2", "numpy"):
    try:
        package_datas, package_binaries, package_hidden = collect_all(package)
    except Exception:
        continue
    datas += package_datas
    binaries += package_binaries
    hidden += package_hidden

a = Analysis(
    [os.path.join(root, "mosaicdiff", "__main__.py")],
    pathex=[root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MosaicDiff",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MosaicDiff",
)
