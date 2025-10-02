import numpy as np

def create_sequences(dataset: np.ndarray, look_back: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of data for time series forecasting.

    Args:
        dataset (np.ndarray): The dataset to create sequences from.
        look_back (int): The number of previous time steps to use as input variables.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the input sequences (X) and output values (Y).
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
