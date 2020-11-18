from setuptools import setup, find_packages


def _require_packages(filename):
    return open(filename).read().splitlines()


long_description = open('README.md', 'r', encoding='utf-8').read()

setup(
    name='emout',
    description='Emses output manager',
    long_description=long_description,
    long_description_content_type="text/markdown",
    version='0.8.3',
    install_requires=_require_packages('requirements.txt'),
    author='Nkzono99',
    author_email='1735112t@gsuite.stu.kobe-u.ac.jp',
    url='https://github.com/Nkzono99/emout',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7'
)
