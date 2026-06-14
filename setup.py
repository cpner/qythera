from setuptools import setup, find_packages
setup(
    name="qythera", version="0.1.0", packages=find_packages(),
    install_requires=["numpy"],
    entry_points={"console_scripts": ["qythera=cli.main:main"]},
    python_requires=">=3.10", author="Qythera Team",
    description="Production Superintelligence Platform", license="MIT",
)
