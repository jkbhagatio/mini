# NLDisco

---

**Ne**ural **L**atent **Disco**very pipeline

## Environment set-up

### With [pixi](https://pixi.sh/latest/tutorials/python) (recommended)

Prerequisites:

- An installed version of [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

- An installed version of [pixi](https://pixi.sh/latest/)

In the root directory, just run `pixi install --manifest-path ./pyproject.toml` - this will create a conda env named 'nldisco'.

### Other

All package dependencies are specified in the 'pyproject.toml'. You can format them as required for your favorite python environment / package management tool, and install them using this tool (e.g. via pip, poetry, conda (directly instead of with pixi), etc.)

## Example usage pipeline

Given:

- Neural data (in the form of binned spike counts as $[examples \times neurons]$)

- Behavioral and/or environmental (meta)data

**NLDisco** performs the following steps to find interpretable neural signatures that underlie behavioral and/or environmental features (referred to collectively as *natural* features)

1. Trains an MSAE to reconstruct the neural data

2. Validates the quality of the MSAE, by looking at

    1. Sparsity of SAE features

    2. Reconstruction quality of neural data

3. Ranks the SAE features by interpretability likelihood

4. (Manual) Finds a corresponding natural feature for each of the top $k$ SAE features.

5. (Manual) Validates the SAE-natural feature pairing by:

    1. Looking at confusion matrix metrics for co-occurrences of the natural feature with the SAE feature.

    2. (TODO) Showing that the neural signature defined by the SAE feature can decode the natural feature well.