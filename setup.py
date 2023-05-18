from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="nprr",
    version="0.1.3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Nonparametric randomized response and locally private confidence sets",
    url="https://github.com/wannabesmith/nprr",
    author="Ian Waudby-Smith",
    author_email="iwaudbysmith@gmail.com",
    license="BSD 3-Clause",
    packages=["nprr"],
    install_requires=["confseq", "numpy", "scipy"],
    zip_safe=False,
)
