from setuptools import setup, find_packages, os

VERSION = '0.0.5'
DESCRIPTION = 'DReAMy'

# with open("README.md", 'r') as f:
#     LONG_DESCRIPTION = f.read()

# Setting up
setup(
    name="dreamy",
    version=VERSION,
    license="Apache-2.0",
    url="https://github.com/lorenzoscottb/DReAMy",
    author="lorenzoscottb (Lorenzo Bertolini)",
    author_email="<lorenzoscottb@gmail.com>",
    description=DESCRIPTION,
    long_description="LONG_DESCRIPTION",
    packages=find_packages(),
    install_requires=["scikit-learn", "transformers[tokenizers,torch]", "tqdm", "pandas", "numpy", "datasets"],
    keywords=['python', 'NLP', 'dream'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3.9",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
    ]
)