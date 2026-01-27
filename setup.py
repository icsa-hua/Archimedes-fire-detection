import setuptools

with open("README.md", "r") as file: 
    long_description = file.read()


with open("requirements.txt", "r") as file: 
    requirements = file.read().splitlines()


setuptools.setup(
    name="GenAI-ICA", 
    version="0.0.1", 
    description="Package to analyze image context data to labels through LLM", 
    long_description=long_description, 
    url="", 
    packages=setuptools.find_packages(include=['src', 'src.*']), 
    install_requires=requirements,
    python_requires='>=3.8'
)
    
