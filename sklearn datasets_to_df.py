import pandas as pd

class sklearn_to_df:
    def sklearn_to_df(self, sklearn_dataset):
        df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
        df['target'] = pd.Series(sklearn_dataset.target)
        return df

