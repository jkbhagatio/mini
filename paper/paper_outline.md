## Things to mention

- Emphasize that MINI:

    - Simple to understand, implement, and use (highlight in paper)

    - Shines in ethological / exploratory settings: "feature discovery" (highlight in paper)

    - Reveals interpretable latents (highlight in paper)
        
        - Latents that are behaviorally or environmentally relevant
        
        - Exact moments when latents are active

    - Yields highly accurate reconstructions from latents (highlight in paper)

    - Is scalable to large datasets (linear time complexity) (mention in paper + show in table)

        - Is computationally cheap

    - Imposes no priors on the latent space (mention in paper + show in table)

    - Can learn complex, nonlinear dynamics (mention in paper + show in table)

    - Can be used with, but does not require, trial-structured data (mention in paper + show in table)

    - Can be used with, but does not require, multimodal data (mention in paper + show in table)

    - Can be used to find and decode continuous and discrete variables (features) (mention in paper + show in table)

    

- How did setting values for `topk` and `d_sae` affect interpretability?

    - How did we go about setting these in general?

- Results Datasets

    - Artificial dataset (e.g. that used in CEBRA paper)

    - Churchland datasets

    - Aeon dataset

- Methods comparisons

    - CEBRA

    - AutoLFADS

    - (d)PCA

    - (sparse/seq)NMF


# Paper (main)

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

    - Extend to multimodal data

        - e.g. reconstruct spikes from spikes + multimodal behavior (e.g. pose-tracking, gaze-tracking, etc.)

    - SCCs

        - Brain diffing

        - Cross-region prediction

    - Sequence length

        - History dependence to improve reconstructions?

Method table comparison:

Here we highlight some key features of MINI and compare it with some other relevant neural LVM methods. These features are:

- *Learns temporally precise latents*: Whether the method can learn latents that are time-locked to neural events at the resolution of the desired event, down to single-spike precision.

- *Reveals discrete and continuous features*: Whether the method can map latents to either discrete (e.g. trial condition) or continuous (e.g. position) real-world features in the data.

- *Incorporates nonlinear dynamics*: Whether the method can use nonlinear neural dynamics when learning latents. 

- *Imposes latent space prior*: Whether the method imposes a prior on the latent space.

- *Requires trial-structured data*: Whether the method requires trial-structured data.

- *Supports trial-structured data*: Whether the method can incorporate trial-structured data, even if not required.

- *Requires multimodal data*: Whether the method requires multimodal data (e.g. video data, or various forms of behavioral data).

- *Supports multimodal data*: Whether the method can incorporate multimodal data, even if not required.

| Method | Learns temporally precise latents | Reveals discrete and continuous features | Incorporates nonlinear dynamics | Imposes latent space prior | Requires trial-structured data | Supports trial-structured data | Requires multimodal data | Supports multimodal data |
|--------|----------------------------------------|---------------------------|----------------------------|-------------------------------|-------------------------------|--------------------------|--------------------------| ---------------------------|
| MINI | Yes | Yes | Yes | No | Yes | No | Yes | No |
| CEBRA | Yes | Yes | No | No | Yes | No | Yes | No |
| AutoLFADS | Yes | Yes | No | No | Yes | No | Yes | No |
| (d)PCA | No | No | Yes | No | Yes | No | Yes | No |
| (sparse/seq)NMF | No | No | Yes | No | Yes | No | Yes | No |

# Paper (supplementary)

## Acknowledgements

## References

## Appendix

### Data and code availability

    - Model and feature viz details

### Additional results

- Allen datasets

### Method table comparison