def analyze_continuous_variable(
    acts_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    variable: str,
    n_bins,  # int OR sequence of edges
    min_activation_frac: float
) -> List[Dict]:
    """Analyzes a continuous variable by binning it and then using the discrete analysis method."""
    is_edges = isinstance(n_bins, (list, tuple, np.ndarray))

    if is_edges:
        try:
            n_bins_eff = len(n_bins) - 1
        except TypeError:
            n_bins_eff = n_bins
        print(f"    Binning '{variable}' into predefined bins ({n_bins_eff} bins)...")
    else:
        print(f"    Binning '{variable}' into {n_bins} bins...")

    binned_col_name = f"{variable}_binned"

    data_to_bin = metadata_binned[variable].dropna()
    if data_to_bin.empty:
        return []

    if is_edges:
        edges = np.asarray(n_bins)
        metadata_binned[binned_col_name] = pd.cut(
            data_to_bin, bins=edges, include_lowest=True
        )
    elif variable == 'movement_angle':
        bins = np.linspace(-180, 180, int(n_bins) + 1)
        metadata_binned[binned_col_name] = pd.cut(
            data_to_bin, bins=bins, include_lowest=True
        )
    else:
        metadata_binned[binned_col_name] = pd.qcut(
            data_to_bin, q=int(n_bins), labels=None, duplicates='drop'
        )

    results = analyze_discrete_variable(acts_df, metadata_binned, binned_col_name, min_activation_frac)

    for res in results:
        res['variable'] = variable
        res['variable_type'] = 'continuous'

    return results


def map_latents_to_metadata(
    acts_df: pd.DataFrame,
    metadata_binned: pd.DataFrame,
    discrete_vars: List[str] = None,
    continuous_vars: List[str] = None,
    min_activation_frac: float = 0.1,
    n_bins_continuous: List = None,  # accepts ints or edge lists per variable
    top_n_mappings: int = 3
) -> pd.DataFrame:  
    """Automatically maps SAE latents to metadata variables, returning a ranked DataFrame of associations."""
    if discrete_vars is None: discrete_vars = []
    if continuous_vars is None: continuous_vars = []

    # Default or broadcast [single] to all continuous variables
    if n_bins_continuous is None:
        n_bins_continuous = [10] * len(continuous_vars)
    elif isinstance(n_bins_continuous, list) and len(n_bins_continuous) == 1 and len(continuous_vars) > 1:
        n_bins_continuous = n_bins_continuous * len(continuous_vars)

    if len(n_bins_continuous) != len(continuous_vars):
        raise ValueError("n_bins_continuous must match length of continuous_vars (or be a single-element list to broadcast).")
    
    all_results = []
    print("Starting automated latent-to-metadata mapping...")

    for variable in discrete_vars + continuous_vars:
        if variable not in metadata_binned.columns:
            raise ValueError(f"Variable '{variable}' not found in metadata_binned")
        print(f"\nAnalyzing variable: {variable}...")
        
        if variable in discrete_vars:
            results = analyze_discrete_variable(acts_df, metadata_binned, variable, min_activation_frac)
            all_results.extend(results)
        elif variable in continuous_vars:
            idx = continuous_vars.index(variable)
            n_bins = n_bins_continuous[idx]  # may be int OR list of edges
            results = analyze_continuous_variable(
                acts_df, metadata_binned, variable,
                n_bins=n_bins, min_activation_frac=min_activation_frac
            )
            all_results.extend(results)
        print(f"    Found {len(results)} potential associations")
    
    if not all_results:
        print("\nNo associations found meeting the minimum activation fraction!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    results_df['value'] = results_df['value'].astype(str)

    print(f"\nSelecting top {top_n_mappings} mappings for each variable/value/instance combination...")
    ranked_df = (
        results_df.sort_values('selectivity_score', ascending=False)
        .groupby(['variable', 'value', 'instance_idx'])
        .head(top_n_mappings)
    )
    
    sort_order = ['variable_type', 'variable', 'value', 'instance_idx', 'selectivity_score']
    ascending_order = [True, True, True, True, False]
    
    final_df = ranked_df.sort_values(by=sort_order, ascending=ascending_order).reset_index(drop=True)
    
    discrete_count = len(final_df[final_df['variable_type'] == 'discrete'])
    continuous_count = len(final_df[final_df['variable_type'] == 'continuous'])
    
    print(f"\nFound {discrete_count} top discrete associations")
    print(f"Found {continuous_count} top continuous associations")
    print(f"Total: {len(final_df)} associations returned in single DataFrame")
    
    return final_df