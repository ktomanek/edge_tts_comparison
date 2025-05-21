from setuptools import setup, find_packages

setup(
    name="tts_lib",
    version="0.1",
    package_dir={"": "src"},  # Look in src/ directory
    packages=find_packages(where="src"),  # Find packages in src/    
)
