#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Luke Lefebure",
    author_email="luke.lefebure@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="An easy to use PyTorch implementation of Dynamic Bernoulli Embeddings",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="dynamic_bernoulli_embeddings",
    name="dynamic_bernoulli_embeddings",
    packages=find_packages(
        include=["dynamic_bernoulli_embeddings", "dynamic_bernoulli_embeddings.*"]
    ),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/llefebure/dynamic_bernoulli_embeddings",
    version="0.1.0",
    zip_safe=False,
)
