import numpy as np


def main():
    """
    Load the equilibrium parameters and print them
    """
    filepath = "toymodel_output/benchmark/equilibrium.npz"
    try:
        data = np.load(filepath)
        print("Loaded parameters:")
        for key in data.files:
            print(f"{key}: {data[key]}")
    except Exception as e:
        print(f"Error loading file: {e}")

        for key in data.files:
            array = data[key]
            if isinstance(array, np.ndarray):
                print(f"{key} (np.ndarray):")
                print(array)


if __name__ == "__main__":
    main()
