from setuptools import setup, find_packages

# import subprocess
# first install pip-tools
# subprocess.run(["pip", "install", "pip-tools"])
# # run pip-compile requirements.in to update requirements.txt
# subprocess.run(["pip-compile", "requirements.in"])

# # Read in the requirements.txt file
# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setup(
    name="collaborative_experiments",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
)
