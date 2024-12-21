import pandas as pd

def load_data(dataset, path, root):
    if (root):
        return pd.read_csv(f"data/{path}/{dataset}")
    return pd.read_csv(f"../data/{path}/{dataset}")


def save_data(dataset, name, path, root):
    if (root):
        dataset.to_csv((f"/data/{path}/{name}.csv"), index=False)
    dataset.to_csv((f"../data/{path}/{name}.csv"), index=False)