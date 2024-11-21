import hashlib

import numpy as np

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # Ensure ID column exists
    if id_column not in df.columns:
        raise ValueError(f"Column {id_column} not found in DataFrame.")

    # Create a numeric hash of each ID if necessary
    if df[id_column].dtype == 'object':
        hash_func = lambda x: int(hashlib.sha256(x.encode()).hexdigest(), 16) % (10 ** 10)
        ids = df[id_column].apply(hash_func)
    else:
        ids = df[id_column].astype(np.int64)

    # Determine the threshold ID value at the specified fraction
    threshold = ids.quantile(training_frac)

    # Assign to 'train' or 'test' based on whether ID is less than or equal to the threshold
    df['sample'] = np.where(ids <= threshold, 'train', 'test')

    return df
    
