def qn(df_input):
    """
    df_input.shape = (n, p),
    where p -- number of genes
    n -- number of probs
    
    """
    df = np.array(df_input.T)
    idx = df.argsort(axis = 0)

    df.sort(axis = 0)
    
    quantiles = np.mean(df, axis = 1)
    idx_final = idx.argsort(axis=0)
    
    res = pd.DataFrame(quantiles[idx_final].T, columns=df_input.columns, index=df_input.index)
    
    return res
