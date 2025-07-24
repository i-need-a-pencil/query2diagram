import setuptools

setuptools.setup(
    name="q2d",
    version="0.0.0",
    description="CodeExplain pipeline",
    packages=setuptools.find_packages(),
    install_requires=[],
    package_data={},
    python_requires=">=3.11",
    entry_points="""
        [console_scripts]
        q2d=q2d.cli:cli
    """,
)
