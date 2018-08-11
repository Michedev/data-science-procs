from distutils.core import setup

setup(
    name='data_science_procs',
    version='1.0',
    packages=['dataset', 'prediction', 'validation', 'visualization'],
    url='https://github.com/Michedev/data-science-procs',
    license='MIT',
    requires=["pandas", 'seaborn', 'matplotlib',
              'numpy', 'scikit-learn', 'scipy'],
    author='mikedev',
    author_email='mik3dev@gmail.com',
    description='Useful functions for data analysis',
    classifiers=(
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Utilities",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only"
    )

)
