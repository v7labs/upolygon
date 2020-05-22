import setuptools
from Cython.Build import cythonize

with open("README.md", "r") as f:
    long_description = f.read()


extensions = [
    setuptools.Extension(
        "draw_polygon", ["upolygon/draw_polygon.pyx"], extra_compile_args=["-O3", "-Wall"]
    )
]

setuptools.setup(
    name="upolygon",
    version="0.1",
    author="V7",
    author_email="simon@v7labs.com",
    description="Collection of fast polygon operations for DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/v7labs/upolygon",
    ext_modules=cythonize(extensions),
    install_requires=[],
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License",],
    python_requires=">=3.6",
)
