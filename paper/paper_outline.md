## Things to mention

- Emphasize that MINI:

    - Simple to understand, implement, and user-friendly (highlight in paper)

    - Shines in ethological / exploratory settings: "feature discovery" (highlight in paper)

        - Gives you a latent ready to explore, in contrast to CEBRA and other methods, which requires you to segment a latent space, for example

            - A difference between mapping individual latents, and mapping areas in the latent space.

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

- Highlight that high-dimensional neural data can often be well approximated by a low-dimensional latent space, which MINI can learn

- Mention looking at neural latents benchmark for methods to compare to

- How did setting values for `topk` and `d_sae` affect interpretability?

    - How did we go about setting these in general?

- Results Datasets

    - Artificial dataset (e.g. that used in CEBRA paper)

    - Churchland datasets

    - Allen datasets (supplementary)

    - Aeon dataset 

- Methods comparisons in main results

    - To do a fair comparison with MINI's main goal of unsupervised interpretable latent discovery, in this work we detail comparisons with paper-published methods that:
        1. Claim an interpretable latent space among their primary goals (as opposed to e.g. only strong decoding performance)
        2. Don't require multimodal nor trial-structured data
        3. Publicly share the application of their method to the Churchland MC_Maze dataset. 
    [Methods Comparisons Table]. Among these, we visualize results from LangevinFlow and CEBRA here in the main text, as at the time of this writing they represent the neural latents benchmark [ref] state-of-the-art for an autoencoder approach and non autoencoder approach, respectively. We also include NMF and PCA as baselines, as they are widely used methods for dimensionality reduction and latent extraction in neural data.

    - Main text comparisons

        - LangevinFlow VAE ([Song et al., 2025a](https://arxiv.org/html/2507.11531v1))
        - CEBRA ([Schneider et al., 2023](https://www.nature.com/articles/s41586-023-06031-6))
        - PCA
        - (sparse)NMF
    
    - Full table comparisons

        - LangevinFlow VAE ([Song et al., 2025a](https://arxiv.org/html/2507.11531v1))
        - CEBRA ([Schneider et al., 2023](https://www.nature.com/articles/s41586-023-06031-6))
        - PCA (Hotelling 1933)
        - (sparse)NMF (Hoyer 2004)
        - hoLDS (Pillow lab)
        - SMC-LR-RNN
        - ST-NDT
        - iLQR-VAE
        - AutoLFADS
        - GPFA

- Comparative analysis

    - Decoding results

    - Training time

    - Inference time

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
    
    - Impose smoothness on latent space

        - Predict next latent space from previous latent space
    
    - Neural latents benchmark scores: compare with other methods

# Paper (supplementary)

## Acknowledgements

## References

## Appendix

### Data and code availability

- Model and feature viz details

### Additional results

- Allen datasets

### Method table comparison

#### Main text comparisons

- LangevinFlow VAE (Song et al., 2025)
- CEBRA (Schneider et al., 2023)
- PCA
- (sparse)NMF

#### Methods table comparisons



#### Other methods to mention

- SIMPL (George et al., 2025)
- MINT (Perkins et al., 2025)
- NCE (Schmutz et al., 2025)
- MtM (Zhang et al. 2024)
- SSMDM (Huang et al., 2024)
- DPAD (Sani et al. 2024)
- sliceTCA (Pellegrino et al., 2024)
- MM-GP-VAE (Gondur et al., 2023)
- NDT2 (Ye et al., 2023)
- TNDM (Hurwitz et al., 2021)
- Ctrl-TNDM
- PLNDE (Kim et al., 2021)
- M-GPL-VM (Jensen et al., 2020)
- VIND (Hernandez et al., 2020)
- MIND (Low et al., 2018)
- pfLDS (Gao et al., 2016)
- PLDS (Macke et al., 2011
