Some notes on things we've tried or have thought about trying. Coud be useful for future reference / paper.


## SAE architectures


- Vanilla (single-hidden layer, ReLU-neurons)

- Multi-layer decoder SAE

    - Applicable if natural features are nonlinearly embedded in the spikes
    
    - Found it qualitatively harder to interpret latents

- Gated

    - Haven't tried, but probably not worth trying as implementation and tuning is a bit tricky, and people report that reconstructions and interpretability are generally worse than with TopK or Jump-ReLU SAEs.

- (Batch)TopK

    - Seems to work well! Nice that we don't have to tune sparsity coefficient.

- Matryoshka SAE (MSAE), or multiple sizes of SAEs

    - Applicable for trying to pull out various "levels" of features

    - We extend MSAEs to use varying topk in addition to varying dsae for the different levels: we find better reconstructions, fewer dead neurons, and better interpretability for higher-level features when using larger topk for these features.
    
    - Using this with batch-topk; awaiting interpretability results!

    - Maybe add weighting to the levels of reconsructions in the loss?

- Jump-ReLU SAE

    - Haven't tried

## SAE loss functions


### Reconstruction Loss


- MSE

    - Seems to work well!

- MSLE

    - Seems to work well! More often than not, it outperforms MSE.

    - `tau` parameter: in most cases when comparing `tau != 1`, penalizing understimations more than overestimations leads to worse performance, and in most cases when including `tau == 1`, this equal weighting of over and underestimations performs best.

- PKL

    - Issues:

        - Difficult to work with given 0-spike data.

            - Can add a small epsilon term, but what this exact value to be is tricky, and then we're being a bit unfaithful to the actual spike data.

        - Seemingly always performed worse than MSE and MSLE.


### Sparsity-penalty loss (for non Batch-TopK and Jump-ReLU SAEs)


- L1-standard

    - Works well, but have to tune sparsity coefficient and explicitly normalize decoder weights on each step.

- L1-decoder-norm

    - Seems to work a little better than L1-standard!

- Tanh

    - Haven't really tried it yet, may not be worth using because of overhead of tuning additional hyperparms (`A` and `B`).

- Sparsity coefficient schedule:

    - Only worth trying if we think this will make l1-loss SAEs better than batch-topk and jump-relu.


## Hyperparameters


### Optimizers


- Learning rate schedule

    - A simple loss-independent cosine cycle seems to work marginally better than static lr, but may not worth be using because of overhead of computing it on each step (though this is fairly minimal) and because of setting/tuning warmup phase, decay phase, and cycle params. <br>...*But, we are using it for now, as it does lead to slightly better reconstruction loss!* With the following settings:

        - `n_warmup_steps = int(n_steps * 0.1)`
        - `decay_start_step = int(n_steps * 0.5)`
        - `n_decay_steps = n_steps - decay_start_step`
        - `n_cycle_steps = int(n_decay_steps * 0.2)`
        - `min_lr = 1e-2 * init_lr`
        - `cycle_amplitude = 0.1`  

    - Probably not worth making loss-dependent, as SAEs often show oscillatory loss patterns as they discover and refine sparse features. e.g. loss can temporarily increase when the SAE "kills" ineffective features to discover better ones. A loss-dependent scheduler might interpret this as "training is diverging" and decrease the rate just as the model is making necessary reorganizations.

    - In general, lr schedules particularly make sense for really long training runs of large models and/or complex loss landscapes. Not to say they can't be used otherwise, but generally none of these hold true for SAEs.


#### Adam variants


- Adam 

    - Seems to work well!
    
- AdamW 

    - Doesn't provide any real benefit in our case because we are not doing any form of weight decay (we don't need to since we enforce sparsity). 

- LAdam

    - Doesn't provide any real benefit in our case because we're using shallow SAEs.

- NAdam: 
    
    - Seems to work marginally better than Adam when properly tuned, but may not be worth using because of overhead of tuning an additional hyperparm (`momentum_decay`).


## Neural data


### Timebin size


Ideally, we want as small as possible, without the data becoming too sparse. 50 ms seems to give about 0.8 sparsity per unit with average unit firing rate of ~ 10 Hz. This seems decent and is similar to 100 ms timebins, so went with this. Could potentially go down to 20 or 10 ms bins if enough unit activity.


### Unit preprocessing


Ideally we'll have clean units. We used ks3.5 labeled good untis and had an additional step of removing neurons with high isi_violations at 2 ms (> 0.1, a sign of contamination), and low firing rates (< 0.5 Hz). We max normalize the spike data.

### Sequence length

Right now, we take one timebin in, and try to reconstruct the same timebin. We could also take multiple timebins in, and try to reconstruct various of these timebins, because longer spiking history may be meaningful for reconstruction / prediction ? Have the code for this and have tested it, but haven't gotten good results with seq_len > 1. Probably worth looking more into.


## Methods to compare vs. MINI


### PCA


### (Sparse) NMF


### LFADS


### CEBRA


### sliceTCA


## General notes


- Ideally we give default SAE hyperparams (or even range of hyperparams for a small sweep) as a function of units and examples in a given dataset.

    - After some initial sweeps over (`d_sae`, `lr`, `lr_sched`, `loss_fn`, and `topk`), we found that `lr_sched=True` and `loss_fn` as some variant of `msle` always outperformed other alternatives for these hyperparams. So we can probably just set these as defaults, and just sweep over `d_sae`, `lr`, and `topk`, and `tau` for msle for given datasets.

    - -> `lr = 5e-3`, `loss_fn = msle1.0` seem to be best performing.

    - How to set `dsae` (heuristic reccomendation)

        - Set Upper limit as `n_examples` / 10.

        - Work down from this upper limit: for a set `top_k`, decrease `dsae` by 10-20%; if mean r2_per_unit and r2_per_exmaple both decrease by more than 5-10%, then repeat, otherwise, select the previous `dsae` size.

    - How to set `topk` (heuristic recommendation)

        - For a given dataset (n_units in animal model brain region(s) R in task T), what is a reasonable number of features (think low-level and high-level) that may be represented in a time window W ?

            - e.g. across mouse visual cortex in a recording with 200 units with Allen visual coding task (natural images), maybe we expect H high-level and L low-level feature representations of the image in a 50 ms window.

- Generally 3 phases that occur during SAE training:

    1. Initial representation: Features are unspecialized, capturing general patterns
    2. Specialization: Features begin to capture specific patterns, sparsity increases
    3. Fine-tuning: Minor adjustments to specialized features for optimal reconstruction


## Other random things


- Rather than reinitializing dead neurons (which is a bit unprincipled and can lead to training instability (loss spikes) and feature collapse), we used an auxiliary loss.


## Allen sessions

- 732592105
- 742951821
- 760693773
