from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="jewelry-enhancement",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade jewelry image enhancement pipeline using Real-ESRGAN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jewelry-enhancement",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jewelry-enhance=scripts.run_enhancement:main",
            "jewelry-degrade=scripts.create_degraded_dataset:main",
            "jewelry-eval=scripts.evaluate_results:main",
        ],
    },
)
