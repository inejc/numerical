from setuptools import setup, find_packages

description = 'Python implementations of various numerical methods.'

with open('README.md') as f:
    long_description = f.read()

setup(
    name='numerical',
    version='0.0.1',
    author='Nejc Ilenic',
    description=description,
    long_description=long_description,
    license='MIT',
    keywords='educational numerical mathematics',
    install_requires=['numpy>=1.12.0', 'scipy>=0.19.0'],
    packages=find_packages(),
    test_suite='tests',
    classifiers=[
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
