"""
The setup.py is an essential part of packaging and distributing Python projects.
It is used to setup tools (or distribution utilities) that are needed to package the project or configure the project such as metadata, dependencies, and more.
"""

from setuptools import setup, find_packages
"""
find_packages() is a function that returns a list of all Python packages found within a directory.
wherever there is a __init__.py file in a directory, that directory will be considered a package.
"""
from typing import List

def get_requirements() -> List[str]:
    
    """
    This function will return list of requirements from requirements.txt file
    """
    
    requirements_list:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            # Read lines from the file
            lines = file.readlines()
            
            # Process each line
            for line in lines:
                
                requirements = line.strip()
                #Ignore the empty lines and -e. from the requirements.txt file
                
                if requirements and requirements!= '-e .':
                    requirements_list.append(requirements)
    
    except FileNotFoundError:
        print("requirements.txt file not found")
        
    return requirements_list 

print(get_requirements())

setup(
    name = "Network Security",
    version = "0.0.1",
    author = "Ganesh Adarkar",
    author_email = 'ganesh.adarkar0811@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements(), 
)