from setuptools import setup, find_packages

setup(
    name="afcluster",
    version="0.1.2",
    description="Cluster Multiple-Sequence Alignments (MSA) using DBSCAN",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Noah Kleinschmidt",
    author_email="noah.kleinschmidt@unibe.ch",
    url="https://github.com/NoahHenrikKleinschmidt/AFCluster-api",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "polyleven",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
