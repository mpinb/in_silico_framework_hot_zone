# Contribute to ISF

Welcome to the ISF contributor guide!

Contents:
- [Code of Conduct](#code-of-conduct)
- [Who can contribute?](#who-can-contribute)
- [How to contribute](#how-to-contribute)
  - [What to consider](#what-to-consider)
  - [Setting up the development environment](#setting-up-the-development-environment)
  - [Coding Standards and Guidelines](#coding-standards-and-guidelines)
  - [Testing](#testing)
  - [Commits](#commits)
  - [Pull Request Process](#pull-request-process)
- [Issue Tracking](#issue-tracking)
- [Documentation](#documentation)

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all. Please read and adhere to our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Who can contribute?

Anyone who uses ISF and has ideas on how to improve or extend the functionality is welcome to contribute!

## How to contribute

### What to consider

Before you start implementing a feature or fixing a bug, please [open an issue](https://github.com/mpinb/in_silico_framework/issues/new/choose) first.
It is not unlikely that some functionality may already be available, or at least possible with ISF. At the same time, not every possibility needs to be part of ISF's source code. 

In the issues tracker, you can also count on the expertise and advice of the developers who have been using and develping ISF for some time now.
That being said, we are also welcome new ideas, and are very excited to hear how you are using ISF!

### Setting up the development environment

ISF uses [`pixi`](https://pixi.sh/latest) for managing environments. The default environment includes the`run dependencies`: everything you need to run ISF.
These are often sufficient to implement new ideas. However, if you require additional dependenices, you can simply `pixi add xyz`. Before adding new dependencies to ISF, please consider the following:

- Dependencies are not always maintained forever. `pandas-msgpack` and `sumatra` have been two examples of packages we had to deprecate or work around in ISF.
- Maintenance costs tend to scale exponentially with additional dependencies
- Can your dependency be reasonably omitted in favor of the standard library, or other core packages such as `numpy` or `scipy`?

### Coding Standards and Guidelines

Please follow these coding standards and guidelines:

- **Code Style**: Follow PEP 8 for Python code.
- **Naming Conventions**: Use descriptive names for variables, functions, and classes.
- **Documentation**: Write docstrings for *all* public functions, classes, and methods.
- **Comments**: Use comments to explain complex logic and important decisions.
- **Formatting**: We prefer the `black` coding style.


### Testing

To test if a new feature works as advertised, we **strongly** recommend you write tests for it. This may not be the flashiest aspect of coding, but untested code can lead to infinitely more hassle and lost time compared to whatever time it takes to write the tests.

While this is not explicit in ISF, you can categorize a test in one of three areas:

- **Unit tests**: Isolated tests that checks if a single object operates as expected under various conditions. Example: what happens if I pass wrong types, negative values ... to a single function?
- **Integration tets**: Broader-scoped tests that check if a piece of software integrates well with existing code. Example: does my new pipeline work well with ISF's existing `data_base` functionality?
- **End-to-end (E2E) tests**: Does my pipeline still operate as expected from start to finish? Example: if I adapt the file format of the cell parameters, can I still run `simrun.run_new_simulations()`?

Running test requires additional dependencies, such as `pytest`. These are defined in our `test` environment. To run the test suite in the test environment, we preconfigured the following command:

```bash
pixi run test
```

To run any other command within the test environment, you can simply prefix the command with:

```bash
pixi run -e test my_command
```

### Commits

Please keep commits single-purpose with descriptive commit messages. Avoid adding all changes in a single monolithic commit.
Write clear and descriptive commit messages. Follow these guidelines:

- **Title**: A short summary of the changes (50 characters or less).
- **Body**: A detailed description of the changes, if necessary. Explain the "why" and "how" of the changes.

### Pull Request Process

To submit a pull request:

1. **Fork the repository** and create your branch from your version of `develop`.
2. **Commit your changes** with clear commit messages.
3. **Push your branch** to your forked repository.
4. **Open a pull request** against ISF's `develop` branch.

In your pull request description, include:

- A summary of the changes.
- Any related issues or pull requests.
- Steps to test the changes.

## Issue Tracking

We use GitHub Issues to track bugs and feature requests. When reporting an issue, please include:

- A clear and descriptive title.
- A detailed description of the problem or request.
- Steps to reproduce the issue, if applicable.
- Any relevant logs, screenshots, or code snippets.

Label your issue appropriately (e.g., `bug`, `enhancement`, `question`).

## Documentation

Our documentation is generated using Sphinx, together with the `autoapi` extension.
Invoking `pixi run build_docs` triggers the Sphinx build process. This is simply a convenience alias for `make html`.
If you are adapting the documentation configuration, you may need to delete the `docs/_build` and `docs/tutorials` directories before rebuilding.

A comprehensive overview of how Sphinx reads in source files (i.e. our Python code) and builds documentation is given at https://www.sphinx-doc.org/en/master/extdev/event_callbacks.html
However, we summarize some key concepts below.

- All configuration (with the exception of templates) is defined in `docs/conf.py`.
- Our documentation uses custom templates, provided in `docs/_templates/autoapi/python`
- When Sphinx writes out documentation, it first writes our "stub pages" that contain the overall structure of the documentation page. These stub pages are reStructuredText (`.rst`) files with structural information of the page, but (generally) no explicit content yet. Only afterwards does it generate HTML (or PDF if you want) from these stub pages.
- Sphinx and reStructuredText heavily rely on directives to make documentation. While there are many extensions with custom directives, this project
  relies on just a handful of core built-in directives: `.. py:obj::` and `.. toctree::`.
- The look and ufnctionality of the documentation website can still *heavily* depend on which HTML theme you are using.
  Often, HTML themes offer very functional extensions beyond what one would consider "just a theme". 
  For example, our current HTML theme (last checked 27/02/2025) is `immaterial`, which ships with a reflinking extension for Graphviz.
- For debugging purposes, you can inspect what the stub files look like in `docs/autoapi` (only if `autoapi_keep_files = True` in `conf.py`)