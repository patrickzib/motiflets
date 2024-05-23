#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ["patrickzib"]

import toml
from setuptools import find_packages, setup
from pathlib import Path

pyproject = toml.load("pyproject.toml")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text("UTF-8")
    
"""Set up package."""
setup(
    author_email=pyproject["project"]["authors"][0]["email"],
    author=pyproject["project"]["authors"][0]["name"],
    classifiers=pyproject["project"]["classifiers"],
    description=pyproject["project"]["description"],
    install_requires=pyproject["project"]["dependencies"],
    include_package_data=True,
    keywords=pyproject["project"]["keywords"],
    license=pyproject["project"]["license"],
    long_description=long_description,
    long_description_content_type='text/markdown',
    name=pyproject["project"]["name"],
    package_data={
        "weasel": [
            "*.csv",
            "*.csv.gz",
            "*.arff",
            "*.arff.gz",
            "*.txt",
            "*.ts",
            "*.tsv",
        ]
    },
    packages=find_packages(
        where=".",
        exclude=["tests", "tests.*"],
    ),
    project_urls=pyproject["project"]["urls"],
    python_requires=pyproject["project"]["requires-python"],
    setup_requires=pyproject["build-system"]["requires"],
    url=pyproject["project"]["urls"]["repository"],
    version=pyproject["project"]["version"],
    zip_safe=False,
)