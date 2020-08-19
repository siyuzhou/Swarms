import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="swarms",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siyuzhou/swarms",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
