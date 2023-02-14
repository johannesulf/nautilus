![Logo](https://raw.githubusercontent.com/johannesulf/nautilus/main/docs/nautilus_text_image.png "Logo")

[![Unit Testing Status](https://img.shields.io/github/actions/workflow/status/johannesulf/nautilus/tests.yml?branch=main&label=tests)](https://github.com/johannesulf/nautilus/actions)
[![Code Coverage](https://img.shields.io/coverallsCoverage/github/johannesulf/nautilus)](https://coveralls.io/github/johannesulf/nautilus?branch=main)
[![Documentation Status](https://img.shields.io/readthedocs/nautilus-sampler)](https://nautilus-sampler.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/nautilus-sampler?color=blue)](https://pypi.org/project/nautilus-sampler/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/nautilus-sampler?color=blue)](https://anaconda.org/conda-forge/nautilus-sampler)
[![License: MIT](https://img.shields.io/github/license/johannesulf/nautilus?color=blue)](https://raw.githubusercontent.com/johannesulf/nautilus/main/LICENSE)
![Language: Python](https://img.shields.io/github/languages/top/johannesulf/nautilus)

Nautilus is an MIT-licensed pure-Python package for Bayesian posterior and evidence estimation. It utilizes importance sampling and efficient space exploration using neural networks. Compared to traditional MCMC and Nested Sampling codes, it needs fewer likelihood calls and produces much larger posterior samples. Additionally, nautilus is highly accurate and produces Bayesian evidence estimates with percent precision.

## Example

This example, sampling a 3-dimensional Gaussian, illustrates how to use nautilus.

```python
import corner
import numpy as np
from nautilus import Prior, Sampler
from scipy.stats import multivariate_normal

prior = Prior()
for key in 'abc':
    prior.add_parameter(key)

def likelihood(param_dict):
    x = [param_dict[key] for key in 'abc']
    return multivariate_normal.logpdf(x, mean=[0.4, 0.5, 0.6], cov=0.01)

sampler = Sampler(prior, likelihood)
sampler.run(verbose=True)
points, log_w, log_l = sampler.posterior()
corner.corner(points, weights=np.exp(log_w), labels='abc')
```

## Documentation

You can find the documentation at [nautilus-sampler.readthedocs.io](https://nautilus-sampler.readthedocs.io).

## License

Nautilus is licensed under the MIT License. The logo uses an image from the Illustris Collaboration.
