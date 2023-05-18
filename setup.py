from setuptools import setup

setup(
    name="nprr",
    version="0.1.2",
    long_description="file: README.md",
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
