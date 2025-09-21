from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autolabel",
    version="1.0.0",
    author="Rex Lee",
    author_email="your.email@example.com",
    description="AutoLabel: Professional video object annotation tool with tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rexlee2/autolabel",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "opencv-contrib-python>=4.8.0",
        "numpy>=1.21.0",
        "typer>=0.9.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
        "detection": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autolabel=autolabel.cli:main",
            "autolabel-validate=validation.cli:main",
        ],
    },
    package_data={
        "autolabel": ["*.yaml"],
    },
    include_package_data=True,
)