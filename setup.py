from setuptools import find_packages, setup
from typing import List

alter_text = "-e ."

def get_requirements(path_to_requirements:str)->List[str]:
    """
    This function reads the necessary requirements
    """
    
    with open(path_to_requirements,"r") as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req!=alter_text]
    return requirements



setup(
    name = "End2End ML Learning",
    version = "0.0.1",
    author = "Shadow",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)

