from setuptools import setup, find_packages

with open('README.md', 'r') as fstream:
    long_description = fstream.read()

setup(
    name='nautilus-sampler',
    version='0.0.1',
    description=('Neural Network Boosted Importance Sampling for Bayesian ' +
                 'Evidence and Posterior Estimation'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='astronomy, statistics',
    url='https://github.com/johannesulf/nautilus',
    author='Johannes U. Lange',
    author_email='julange.astro@pm.me',
    include_package_data=True,
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'scikit-learn', 'tqdm'],
    extras_require={'tensorflow': ['tensorflow']},
    python_requires='>=3.0'
)
