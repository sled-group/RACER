from setuptools import setup, find_packages

requirements = [
    "numpy",
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
    "clip @ git+https://github.com/openai/CLIP.git",
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
