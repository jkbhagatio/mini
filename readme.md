# mini

---

**m**echanistic **i**nterpretability for **n**eural **i**nterpretability

## Environment set-up

### With [pixi](https://pixi.sh/latest/tutorials/python) (recommended)

Prerequisites:

- An installed version of [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

In the root directory, just run `pixi install --manifest ./pyproject.toml` - this will create a conda env named 'mini'.

### Other

All package dependencies are specified in the 'pyproject.toml'. You can format them as required for your favorite python environment / package management tool, and install them using this tool (e.g. via pip, poetry, conda (directly instead of with pixi), etc.)

## Example usage pipeline

Given:

- Neural data (in the form of binned spike counts as $[examples \times neurons]$)

- Behavioral and/or environmental metadata

**mini** performs the following steps to find interpretable neural signatures that underlie behavioral and/or environmental features (referred to collectively as *natural* features)

1. Splits the neural data into train/val/test sets

2. Trains an SAE (on the train + val splits) to reconstruct the neural data

3. Validates the quality of the SAE, by looking at

    1. Sparsity of SAE features

    2. Reconstruction quality of neural data

4. Ranks the SAE features by interpretability likelihood

5. For each of the top $k$ SAE features, finds a corresponding natural feature with manual feedback.

6. Validates the SAE-natural feature pairing on the test split, by:

    1. Looking at confusion matrix metrics for co-occurrences of the natural feature with the SAE feature.

    2. Showing that the neural signature defined by the SAE feature can decode the natural feature as well as it can be decoded by the entirety of the neural data.