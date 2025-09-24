from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
      requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tts_lib",
    version="0.1",
    description="A library for conveniently running different tts models, focus is on models that can run locally.",
    author="Katrin Tomanek",
    author_email="katryn.tomanek@gmail.com",    
    url="https://github.com/ktomanek/edge_tts_comparison",    
    package_dir={"": "src"},  # Look in src/ directory
    packages=find_packages(where="src"),  # Find packages in src/   
    install_requires=requirements,
    python_requires=">=3.10", 
)
