![Logo](https://raw.githubusercontent.com/johannesulf/nautilus/main/docs/source/nautilus_text_image.png "Logo")

[![PyPI](https://img.shields.io/pypi/v/nautilus-sampler?color=blue)](https://pypi.org/project/nautilus-sampler/)
[![Documentation Status](https://img.shields.io/readthedocs/nautilus-sampler)](https://nautilus-sampler.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/github/license/johannesulf/nautilus?color=blue)](https://raw.githubusercontent.com/johannesulf/nautilus/main/LICENSE)
![Language: Python](https://img.shields.io/github/languages/top/johannesulf/nautilus)

`nautilus` is an MIT-licensed pure-Python package for Bayesian posterior and
evidence estimation. It is based on importance sampling and efficient space
tessellation using neural networks. Its main features are computational
efficiency as well as accuracy of the posterior and evidence estimates.

## Example

This simple example, sampling a 3-dimensional Gaussian, illustrates how
`nautilus` is used.

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

sampler = Sampler(prior, likelihood, n_live=500)
sampler.run(verbose=True)
points, log_w, log_l = sampler.posterior()
corner.corner(points, weights=np.exp(log_w), labels='abc')
```

## Documentation

You can find the documentation at [nautilus-sampler.readthedocs.io](https://nautilus-sampler.readthedocs.io).

## License

`nautilus` is licensed under the MIT License. The logo uses an image from the
Illustris Collaboration.
