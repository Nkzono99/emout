from setuptools import setup, find_packages


def _require_packages(filename):
    return open(filename).read().splitlines()


setup(
    name='emout',
    description='Emses output manager',
    version='0.5.0',
    install_requires=_require_packages('requirements.txt'),
    author='Nkzono99',
    author_email='1735112t@gsuite.stu.kobe-u.ac.jp',
    url='https://github.com/Nkzono99/emout',
    packages=find_packages()
)
