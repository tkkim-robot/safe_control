from setuptools import setup, find_packages

setup(
    name='cbf_tracking',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'cvxpy',
        'numpy==1.26.4',  # latest gurobipy version 11.0.2 is not compatible with numpy 2.0, but 11.0.3 will be
        'matplotlib',
        'gurobipy',
        'shapely',
        'do-mpc[full]'
    ],
    include_package_data=True,
    zip_safe=False,
)