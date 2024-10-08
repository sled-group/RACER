from setuptools import setup, find_packages

requirements = [
    "numpy==1.24.3",
    "click",
    "scipy",
    "einops",
    "pyrender",
    "transformers",
    "omegaconf",
    "natsort",
    "cffi",
    "pandas",
    "tensorflow==2.13.1",
    "pyquaternion",
    "matplotlib",
    "numpy-quaternion==2023.0.2",
    "clip @ git+https://github.com/openai/CLIP.git",
    "fastapi==0.111.0"
]

__version__ = "0.0.1"
setup(
    name="racer",
    version=__version__,
    description="RACER",
    long_description="",
    author="Yinpei Dai",
    author_email="daiypl@umich.edu",
    url="",
    keywords="robotics,language",
    packages=['racer'],
    install_requires=requirements,
)
