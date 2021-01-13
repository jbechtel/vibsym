import setuptools

VERSION = "0.0.0"

REQUIRED_PACKAGES = []
with open("requirements.txt", "r") as reqs_txt_file:
    REQUIRED_PACKAGES = [line.strip() for line in reqs_txt_file]

REQUIRED_DEV_PACKAGES = []
with open("requirements-dev.txt", "r") as reqs_dev_txt_file:
    REQUIRED_DEV_PACKAGES = [line.strip() for line in reqs_dev_txt_file]

setuptools.setup(
    name="vibsym", 
    version=VERSION,
    author="Jonathon Bechtel",
    description="A small symmetry package for 2D molecules",
    url="https://github.com/jbechtel/vibsym",
    packages=setuptools.find_packages(),
    package_dir={"": "src"},
    python_requires='==3.7.2',
)
