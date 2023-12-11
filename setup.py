from setuptools import setup

setup(
    name='pyBAKS',
    version='0.1.4',
    description='python implementation of Bayesian Adaptive Kernel Smoothing',
    author='Daniel Svedberg',
    author_email='dan.ake.svedberg@gmail.com',
    url='https://github.com/danielsvedberg/pyBAKS',
    packages=['pyBAKS'],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
    ],
)

