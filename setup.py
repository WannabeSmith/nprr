from setuptools import setup

setup(
    name="nprr",
    version="0.1",
    description="Nonparametric randomized response and locally private confidence sets",
    url="github.com/wannabesmith/nprr",
    author="Ian Waudby-Smith",
    author_email="iwaudbysmith@gmail.com",
    license="BSD 3-Clause",
    packages=["nprr"],
    install_requires=[
        "confseq",
        "numpy",
        "scipy",
        "matplotlib"
    ],
    zip_safe=False,
)
