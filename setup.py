import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='qmagen',
    version='1.0.0',
    author='Sizhuo YU.',
    author_email='yusizhuo@buaa.edu.cn',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    # package_dir={'': ''},
    packages=setuptools.find_packages(where=''),
    install_requires=['torch', 'bayesian-optimization', 'sklearn', 'matplotlib', 'dill'],
    python_requires='>=3.6, <4',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3.0',
        'Operating System :: OS Independent',
    ],
    project_urls={})
