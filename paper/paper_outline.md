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

# Paper (main)

## Introduction

### Latent space vs. latent interpretables

- We don't need to explicitly model the entire latent space to find interpretable latents

- All dynamical systems are Markovian when the state space is appropriately augmented. Non-Markovianity is often a property of the observer's representation, not of the system itself.

- Takens' theorem: you can reconstruct a dynamical system's attractor from time-lagged observations

- Perhaps usefulness of non-Markovian methods (e.g., that include explicit recurrence or spike history) is just due to our incomplete or coarse-grained observations of brain state.

    - In recurrent neural networks, the hidden state is updated recurrently — but the forward dynamics are Markovian in the hidden state.

    - When modeling spike trains with history filters (Pillow et al. 2008), we augment the state to improve predictive performance. But this is conceptually the same as saying: "We’re trying to approximate the brain’s internal Markov state using externally observable quantities."

- Related Perspectives in the Literature

  - Churchland et al. (2012) (Neural population dynamics during reaching) argue that neural dynamics in motor cortex can be modeled as a low-dimensional dynamical system — implicitly Markovian in the population state.

  - Sussillo & Barak (2013) (Opening the black box: low-dimensional dynamics in high-dimensional RNNs) show how RNNs learn internal state trajectories that can be Markovian even if the input/output behavior looks non-Markovian.

  - Predictive coding and Friston's free energy principle work assumes the brain maintains sufficient internal beliefs (states) to make predictions, which again implies a Markovian internal state process.

### Related work

## Methods

### Model

- SAEs

- MSAEs

### Pipeline

(Spike sorter output -> feature extraction)

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
