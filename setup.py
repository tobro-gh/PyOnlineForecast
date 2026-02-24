# !/usr/bin/env python

from setuptools import setup, find_packages

with open("py_online_forecast/requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name='py_online_forecast',
    packages=find_packages(),
    version='0.0.1',
#    description='Description...'
#    author='Author',
#    license='MIT',
#    author_email='...',
    package_dir={'': '.'},
    py_modules=['core', 'hierarchies'],
    python_requires='>=3.12.3',
    install_requires=install_requires,
    include_package_data=True,
    package_data={
        '': ['requirements.txt'],
    },
)
