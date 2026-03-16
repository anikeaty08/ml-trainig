from setuptools import find_packages, setup


setup(
    name="ml-pipeline-agent",
    version="0.1.0",
    description="Local-first autonomous ML pipeline agent",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "fastapi>=0.115,<1.0",
        "uvicorn[standard]>=0.30,<1.0",
        "python-multipart>=0.0.9",
        "pandas>=2.2,<3.0",
        "numpy>=2.0,<3.0",
        "scikit-learn>=1.5,<2.0",
        "scipy>=1.13,<2.0",
        "jinja2>=3.1,<4.0",
        "joblib>=1.4,<2.0",
        "httpx>=0.27,<1.0",
        "Pillow>=10.4,<12.0",
    ],
)
