"""
setup.py — Package setup for the trading_machine project.
"""

from setuptools import setup, find_packages

setup(
    name="trading_machine",
    version="1.0.0",
    description="Fully autonomous, self-learning trading machine",
    author="Trading Machine",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "torchaudio>=2.2.0",
        "stable-baselines3>=2.2.1",
        "gymnasium>=0.29.1",
        "streamlit>=1.29.0",
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "plotly>=5.18.0",
        "requests>=2.31.0",
        "sqlalchemy>=2.0.23",
        "tables>=3.9.2",
        "pyarrow>=14.0.1",
        "apscheduler>=3.10.4",
        "python-dotenv>=1.0.0",
        "openpyxl>=3.1.2",
        "scikit-learn>=1.3.2",
        "scipy>=1.11.4",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "tqdm>=4.66.1",
        "pydantic>=2.5.2",
        "loguru>=0.7.2",
    ],
    entry_points={
        "console_scripts": [
            "trading-machine=run:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Trading",
        "Programming Language :: Python :: 3.12",
    ],
)
