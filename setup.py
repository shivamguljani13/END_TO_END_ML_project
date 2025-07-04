from setuptools import setup, find_packages
from typing import List


HYPEN = '-e .'
def get_requirements(file_path: str) -> List[str]:
    requirements=[]
    with open('requirements.txt') as file:
        requirements = file.readlines()
        requirements = [x.replace("\n","") for x in requirements]
        
        if HYPEN in requirements:
            requirements.remove(HYPEN)
    return requirements






setup(
    name='mlproject_requirements',
    version='0.1',
    author='Shivam Guljani',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    
    )