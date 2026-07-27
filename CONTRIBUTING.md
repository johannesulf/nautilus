# Contributing to ``nautilus``

We are excited about contributions to ``nautilus`` from community members. Below, we outline the general workflow and coding guidelines.

## Workflow

Code contributions are submitted via pull requests on GitHub. To do this, you will **fork** ``nautilus`` and create a feature branch that contains your code contributions. Once you are satisfied, create a pull request against the ``main`` branch of <https://github.com/johannesulf/nautilus>. A package maintainer will then work with you on your proposed changes.

## Guidelines

As an open science package, we want to ensure that ``nautilus`` remains well-tested, documented, reliable, and easy to modify. Thus, we ask that code contributions meet the following basic guidelines. All commands assume that you are in the main ``nautilus`` directory.

### Installation

Please check that your modified version of ``nautilus`` installs correctly via:

```
pip install -e .
```

In the unlikely case your modifications require updated dependencies, please list those in the ``pyproject.toml`` file.

### Unit Tests

We use ``pytest`` and GitHub Actions for continuous integration. Once you create a pull request, GitHub Actions will automatically trigger unit tests. Those are required to pass before your contributions can be accepted. To perform the unit tests locally, run:

```
pytest tests
```

If your code contributions add new features, we also encourage you to add unit tests under ``tests`` that verify they work as expected.

### Docstrings

Please ensure that all public functions you contribute have valid [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

### Syntax

Please ensure that the modified code complies with the PEP8 syntax standard. GitHub Actions uses ``ruff`` to test for this and your code contributions can only be accepted after they comply with PEP8. To run the syntax check locally, from the main ``nautilus`` directory, run:

```
ruff check nautilus
```

### Generative AI

If you used generative AI to help you draft your code contributions, you must disclose that in the pull request. Additionally, if new code was substantially written by AI, you must outline what steps you have taken to ensure that the code works correctly.

Generally, we discourage community members from contributing code that they could not have written without AI since, in this case, they may not be able to verify its accuracy. On the other hand, using AI to check existing code or to copyedit documentation can be a good idea.