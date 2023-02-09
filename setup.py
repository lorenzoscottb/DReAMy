from setuptools import setup, find_packages, os

VERSION = '0.0.4'
DESCRIPTION = 'DReAMy'
LONG_DESCRIPTION = 'A package for Dream-Reports Annotation Methods with python.'

# Setting up
setup(
    name="dreamy",
    version=VERSION,
    author="lorenzoscottb (Lorenzo Bertolini)",
    author_email="<lorenzoscottb@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
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