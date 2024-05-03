from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .'
def get_requirements(filename):
    """
    This function returns a list of requirements from a requirements.txt file.
    """
    requirements = []
    with open(filename, 'r') as file_object:
        requirements = file_object.readlines()
        requirements = [requirement.replace("\n","") for requirement in requirements]

        if HYPHEN_E_DOT in requirements: # to remove the -e. from the new requires.txt file that is created by pip
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements
        
setup(
    name = 'mlproject',
    version = '0.01',
    author = "Mauro Small",
    author_email ="mauro.small@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)

