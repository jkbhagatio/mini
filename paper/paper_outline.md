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

    - Can be used with different modalities of neural data (e.g. spike, 2p, LFP)

- How did setting values for `topk` and `d_sae` affect interpretability?

    - How did we go about setting these in general?

- Results Datasets

    - Artificial dataset (e.g. that used in CEBRA paper)

      - Use RatInABox or NeuralPlayground or OurOwnMethod to construct ground-truth interpretable latents, and show that MINI can find them all
    
        - If can find all or almost all, then we're good with this alone! If not, then we need to show comparison to other methods to show it's not significantly worse. First option is ideal!

    - Churchland datasets

    - Allen datasets (supplementary)

    - Aeon dataset 

- Methods comparisons in main results

    - To do a fair comparison with MINI's main goal of unsupervised interpretable latent discovery, we detail comparisons [Methods Comparisons Table] with paper-published methods that:
        1. Claim an interpretable latent space among their primary goals (as opposed to e.g. only strong decoding performance)
        2. Don't require multimodal nor trial-structured data
        3. Publicly share the application of their method to the Churchland MC_Maze dataset. 
    
    Among these, we visualize results from LangevinFlow and CEBRA, as they represent current neural latents benchmark [ref] state-of-the-art methods for an autoencoder approach and non autoencoder approach, respectively. We also include NMF and PCA as baselines, as they are widely used methods for dimensionality reduction and latent extraction in neural data.

    - Main text comparisons

        - LangevinFlow VAE ([Song et al., 2025a](https://arxiv.org/html/2507.11531v1))
        - CEBRA ([Schneider et al., 2023](https://www.nature.com/articles/s41586-023-06031-6))
        - PCA
        - (sparse)NMF
    
    - Full table comparisons

        - PCA (Hotelling 1933)
        - (sparse)NMF (Hoyer 2004)
        - LangevinFlow VAE ([Song et al., 2025a](https://arxiv.org/html/2507.11531v1))
        - CEBRA ([Schneider et al., 2023](https://www.nature.com/articles/s41586-023-06031-6))
        - ST-NDT
        - AutoLFADS

- Comparative analysis

    - Finding interpretable latents

    - Decoding results

    - Training/Inference time

---

# Figures

0. Intro figure
    - Neural data -> explicit latent space vs. extracting interpretable latents
    - (Optional to include, but ideally yes, space permitting)

1. Pipeline

2. Our model arch: MSAE variant

---

# Paper (main)

## Introduction

- Motivation

  - Need to interpret large-scale neural recordings

  - Need dimensionality reduction / LVMs to do this
  
  - Existing methods have many limitations, particularly in regards to interpretability of latent spaces

- How MINI addresses these limitations + can be used with virtually any neural recording modality

- Latent space vs. interpretable latents overview: we just care about interpretable latents

## Methods

### Pipeline Overview

(Spike sorter output -> feature extraction)

- Data preprocessing

- Model training

  - Hyperparameters and sweeps

- Model evaluation

- Feature evaluation

### Model Architecture

- SAEs

- MSAEs

## Results

- Ref method comparisons table

- Discrete and continuous, environmental and behavioral features
- Same animal, cross-session: same features
- Hierarchical features
- Comparing found latents with other methods

### Synthetic (RatInABox?) dataset

- True features

    - 8 total

        - High pos velocity point1, high neg velocity point1, high pos velocity point2, high neg velocity point2, pos1, pos2, pos3, pos4

        - Mapping of each of these features to individual latents

        - Decoding position and velocity from latents alone

    - Made from two pairs of single-cell place cells with slightly overlapping place-fields

### Churchland datasets

- Highlighting a few found features (latents)

- Comparing found latents with other methods

- Decoding results from SAE features

### Allen Neuropixels visual coding datasets

- Highlighting a few found features (latents)

- Comparing found latents with other methods

- Decoding results from SAE features

### Aeon dataset

- Highlighting a few found features (latents)

- Decoding results from SAE features

## Discussion

- Summary

- Pros & Cons / Limitations

  - - Interpretable latents in 2 regards:
  - feature they correspond to
  - neurons that contribute to them
    - Real-time 'steering': opto of cells that contribute to a specific latent

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

---

# Paper (supplementary)

## Acknowledgements

## References

## Appendix

### Code and data availability

- Model and feature viz details

### Additional model details

### Additional results

- Allen datasets

### Method comparisons

- Method comparisons table

- Method comparisons table details


---

- SIMPL (George et al., 2025)
- NCE (Schmutz et al., 2025)
- MINT (Perkins et al., 2024)
- SMC-LR-RNN (Pals et al., 2024)
- MtM (Zhang et al. 2024)
- MM-GP-VAE (Gondur et al., 2024)
- DPAD (Sani et al. 2024)
- sliceTCA (Pellegrino et al., 2024)
- NDT2 (Ye et al., 2023)
- iLQR-VAE (Schimel et al., 2022)
- TNDM (Hurwitz et al., 2021)
- Ctrl-TNDM (Kudryashova et al., 2023)
- PLNDE (Kim et al., 2021)
- M-GPL-VM (Jensen et al., 2020)
- VIND (Hernandez et al., 2020)
- MIND (Low et al., 2018)
- pfLDS (Gao et al., 2016)
- PLDS (Macke et al., 2011)
- GPFA (Yu et al., 2009)
