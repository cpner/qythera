from setuptools import setup, find_packages

setup(
    name="qythera",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "click",
        "rich",
        "requests",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "qythera=cli.main:main",
        ],
    },
    python_requires=">=3.10",
    author="Qythera Team",
    description="Production Superintelligence Platform",
    license="MIT",
)
