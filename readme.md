# NLDisco

**Ne**ural **L**atent **Disco**very pipeline

---

## Environment set-up

### With [pixi](https://pixi.sh/latest/tutorials/python) (recommended)

Prerequisites:

- An installed version of [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

- An installed version of [pixi](https://pixi.sh/latest/)

In the root directory, just run `pixi install --manifest-path ./pyproject.toml` - this will create a conda env named 'nldisco'.

### Other

All package dependencies are specified in the 'pyproject.toml'. You can format them as required for your favorite python environment / package management tool, and install them using this tool (e.g. via uv, pip, poetry, conda, etc.)
