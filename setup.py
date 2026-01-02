from setuptools import setup, find_packages

setup(
    name='opentelevision',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'urdf-parser-py',
        'numpy',
        'scipy',
        'vuer',
    ],
    description='Open-TeleVision: Teleoperation with Immersive Active Visual Feedback',
    author='OpenTelevision Team',
    url='https://github.com/adityabhas22/OpenTelevision',
)
