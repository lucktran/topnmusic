from setuptools import setup


setup(
    name='topnmusic',
    version='0.1',
    description='A music genre classifier',
    packages=['src'],
    install_requires=[
        'librosa',
        'matplotlib',
        'numpy',
    ],
)
