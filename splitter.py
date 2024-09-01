from typing import List
import pandas as pd
import os
from pandas import DataFrame
import random
from tqdm import tqdm


def create_splits(df: DataFrame, label_column: str, proportions: list = None) \
        -> List[DataFrame]:
    if proportions is None:
        proportions = [0.60, 0.20, 0.20]

    orig_columns = df.columns

    if len(df) == 0:
        print('warning: dataframe was empty.')
        return [DataFrame([], columns=orig_columns)] * len(proportions)

    # group by labels and shuffle the groups
    grouped = df.groupby(label_column)
    groups = [grouped.get_group(g) for g in grouped.groups]
    random.shuffle(groups)

    # concatenate the shuffled groups
    sdf = pd.concat(groups).reset_index(drop=True)

    splits = []

    prop = 0
    curr_idx = 0
    for split_prop in tqdm(proportions, 'splitting data'):
        split_df = DataFrame([], columns=orig_columns)
        prop += split_prop
        end_idx = int(prop * len(sdf[label_column]))

        # ensure that there are no common labels between each split
        while curr_idx < end_idx:
            curr_label = sdf[label_column][curr_idx]
            group_start = curr_idx

            while curr_idx < end_idx and curr_label == sdf[label_column][curr_idx]:
                curr_idx += 1

            group_end = curr_idx

            split_df = split_df._append(sdf.iloc[group_start:group_end])

        splits.append(split_df)

    return splits

if __name__ == '__main__':
    dir_prefix = r'./DELETION-INSERTION-MULTIPLE-REPLACEMENT-SINGLE/'
    df = pd.read_csv(os.path.join(dir_prefix, 'all_data_with_context.csv'))
    train_df, val_df, test_df = create_splits(df, 'target_sentence')
    train_df.to_csv(''.join([dir_prefix, 'train_data.csv']), index=False)
    val_df.to_csv(''.join([dir_prefix, 'val_data.csv']), index=False)
    test_df.to_csv(''.join([dir_prefix, 'test_data.csv']), index=False)
