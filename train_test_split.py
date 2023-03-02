
import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_and_save_datasets():

    parser = argparse.ArgumentParser(description='Demo script')
    parser.add_argument('-i', type=str, help='input dataframe path', dest='input_data')
    parser.add_argument('-o', type=str, help='specific output dir', dest='output_dir', default=None)
    parser.add_argument('-test-share', type=float, help='dataset test share', dest='test_share', default=0.2)
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False, help='verbose dataset split')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    base_path = os.path.join(os.environ.get('ROOT_PATH'))

    df = pd.read_csv(os.path.join(base_path, args.input_data))

    if args.output_dir is not None:
        save_path = os.path.join(base_path, args.output_dir)
    else:
        save_path = os.path.dirname(os.path.join(base_path, args.input_data))

    logging.info(f'Original dataset: {len(df)}')
    df = df.drop_duplicates()
    logging.info(f'Final dataset: {len(df)}')

    df_train_val, df_test = train_test_split(df, test_size=args.test_share, random_state=42)
    df_train, df_val = train_test_split(df_train_val, test_size=args.test_share, random_state=42)

    logging.info(f'Train dataset: {len(df_train)}')
    logging.info(f'Valid dataset: {len(df_val)}')
    logging.info(f'Test dataset: {len(df_test)}')

    df_train.to_csv(os.path.join(save_path, 'train_df.csv'), index=False)
    df_val.to_csv(os.path.join(save_path, 'valid_df.csv'), index=False)
    df_test.to_csv(os.path.join(save_path, 'test_df.csv'), index=False)
    logging.info('Datasets successfully saved!')


if __name__ == '__main__':
    split_and_save_datasets()
