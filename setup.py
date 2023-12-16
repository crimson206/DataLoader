from setuptools import setup, find_packages

setup(
    name="DataLoader",
    version="0.1.2",
    description="Personal DataLoader Module",
    author="Sisung Kim",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
    ],
)
