#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Ehsan Shaghaei",
    author_email='ehsan2754@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This project is implementation of multiple AI agents based on different Reinforcement Learning methods  to OpenAI Gymnasium Lunar-Lander environment which is classic rocket trajectory optimization problem.",
    entry_points={
        'console_scripts': [
            'lunarlander_gym=lunarlander_gym.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lunarlander_gym',
    name='lunarlander_gym',
    packages=find_packages(include=['lunarlander_gym', 'lunarlander_gym.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ehsan2754/lunarlander_gym',
    version='0.1.0',
    zip_safe=False,
)
