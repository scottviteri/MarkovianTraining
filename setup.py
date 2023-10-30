from setuptools import setup, find_packages, find_namespace_packages

# import subprocess
# first install pip-tools
# subprocess.run(["pip", "install", "pip-tools"])
# # run pip-compile requirements.in to update requirements.txt
# subprocess.run(["pip-compile", "requirements.in"])


# # Read in the requirements.txt file
# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()
def read_requirements():
    with open("requirements.txt", "r") as file:
        return [line.strip() for line in file if line.strip()]


setup(
    name="collaborative_experiments",
    version="0.1",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
)
