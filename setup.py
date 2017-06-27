from distutils.core import setup

setup(
    name='DeepRL',
    version='0.0.1',
    author='hantian.pang',
    packages=[
        'DeepRL',
        'DeepRL/Agent',
        'DeepRL/Replay',
        'DeepRL/Train',
    ],
    install_requires=[
        'numpy',
        'gym',
    ]
)
