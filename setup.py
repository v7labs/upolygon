import setuptools
import os

with open("README.md", "r") as f:
    long_description = f.read()

USE_CYTHON = os.getenv("USE_CYTHON") in ["TRUE", "true"]
ext = ".pyx" if USE_CYTHON else ".c"

extensions = [
    setuptools.Extension(
        "draw_polygon",
        ["upolygon/draw_polygon" + ext],
        extra_compile_args=["-O3", "-Wall"],
    ),
    setuptools.Extension(
        "find_contours",
        ["upolygon/find_contours" + ext],
        extra_compile_args=["-O3", "-Wall"],
    ),
    setuptools.Extension(
        "simplify_polygon",
        ["upolygon/simplify_polygon" + ext],
        extra_compile_args=["-O3", "-Wall"],
    ),
    setuptools.Extension(
        "run_length_encoding",
        ["upolygon/run_length_encoding" + ext],
        extra_compile_args=["-O3", "-Wall"],
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    extensions = cythonize(extensions)

setuptools.setup(
    name="upolygon",
    version="0.1.7",
    author="V7",
    author_email="simon@v7labs.com",
    description="Collection of fast polygon operations for DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/v7labs/upolygon",
    ext_modules=extensions,
    install_requires=["numpy"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
