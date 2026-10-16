from setuptools import setup, find_packages

setup(
    name="inferx",
    version="1.0.0",
    description="Enterprise Intelligent AI Inference Optimization & Serving Platform",
    author="InferX Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "onnxruntime>=1.15.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
        "streamlit>=1.25.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tabulate>=0.9.0",
        "onnx>=1.15.0",
        "onnxscript>=0.1.0",
    ],
    entry_points={
        "console_scripts": [
            "inferx=inferx.cli:main",
        ],
    },
)
