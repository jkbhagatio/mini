## Things to mention

- How did setting values for `topk` and `d_sae` affect interpretability?

    - How did we go about setting these in general?


# Full paper outline

## Introduction

- Related work

## Methods

### Model

- SAEs

- MSAEs

### Pipeline

- Neural data preprocessing

- Training model

  - Hyperparameters and sweeps

- Evaluating model

- Building dashboard for feature hunting

- Evaluating features

## Results

### Allen Neuropixels visual coding datasets

- Highlighting a few found features (latents)

- Comparing found latents with other methods

- Decoding results from SAE feature spikes

### Churchland datasets

- Highlighting a few found features (latents)

- Comparing found latents with other methods

- Decoding results from SAE feature spikes

### Aeon dataset

- Highlighting a few found features (latents)

- Decoding results from SAE feature spikes

## Discussion

- Summary

- Pros / Cons

- Future work

    - Extend to other recording modalities

        - LFP & EEG (reconstruct voltage values from C channels over T time) 
        
        - Calcium imaging (reconstruct dF/F values from R RoIs over T time)

    - SCCs

        - Brain diffing

        - Cross-region prediction

    - Sequence length

        - History dependence to improve reconstructions?

## Acknowledgements

## References

## Appendix