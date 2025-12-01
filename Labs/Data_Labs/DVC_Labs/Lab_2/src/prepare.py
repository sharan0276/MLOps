import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def prepare_data(input_path="data/penguins_raw.csv", output_path="data/penguins_clean.csv"):
    """
    Clean the data set ensuring same case for columns in dataset and remove null values

    Args :
        - input_path : Path to raw dataset

    Output:
        - output_path : Saved Path for updated / clean  dataset
    """
    logger.info("Preparing dataset for Version 2...")

    df = pd.read_csv(input_path)

    # Version 2 cleaning
    df = df.rename(columns=str.lower)
    df["species"] = df["species"].str.lower()
    df = df.dropna()

    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to {output_path}")

if __name__ == "__main__":
    prepare_data()
