import pandas as pd


class Syntdator:
    def generate(
        self, dist_mapping: dict = None, data: pd.DataFrame = None, size: int = 0
    ) -> pd.DataFrame:
        """Abstract method for generating a synthetic dataframe

        Args:
            dist_mapping (dict): mapping of distributions
            data (pd.DataFrame): used for automate generation of dist_mapping

        Returns:
            pd.DataFrame: synthetic data
        """

        if dist_mapping is not None:
            return pd.DataFrame(
                {k: v.sample(count=size) for k, v in dist_mapping.items()}
            )

        return pd.DataFrame()
