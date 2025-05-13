import importlib
import pkg_resources

required = {
    "torch": None,
    "torchvision": None,
    "opencv-python": None,
    "numpy": None,
    "matplotlib": None,
    "Pillow": None,
    "scipy": None,
    "pyyaml": None,
    "tqdm": None,
    "seaborn": None,
    "pandas": None,
    "tensorboard": None,
    "scikit-learn": None,
    "cython": None,
    "pycocotools": None,
    "requests": None,
    "pywin32": None,
    "onnx": None,
    "onnxruntime": None,
    "thop": None,
    "psutil": None
}

print("Checking installed packages:\n")
for pkg in required:
    try:
        module = importlib.import_module(pkg.replace("-", "_"))
        version = pkg_resources.get_distribution(pkg).version
        print(f"{pkg:<20} ✅ Installed (version: {version})")
    except ImportError:
        print(f"{pkg:<20} ❌ Not installed")
    except Exception as e:
        print(f"{pkg:<20} ⚠️ Error: {str(e)}")
