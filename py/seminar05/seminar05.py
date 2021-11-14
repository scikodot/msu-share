import os
import pandas as pd

def nans(df):
    nans = df.isna().agg('sum', axis=1).astype(bool)
    return df.drop(df.index[nans].tolist())

def av_agent(df):
    av_agent = df['OBJECT_TYPE'].isin(['AV', 'AGENT'])
    return df.drop(df.index[av_agent].tolist())

def shorts(df):
    # Group by ID
    tracks = df[['TRACK_ID', 'X', 'Y']].groupby('TRACK_ID').count()

    # Get indexer
    cond = tracks['X'] <= 10

    # Extract
    ids = tracks.index[cond].tolist()
    shorts = df['TRACK_ID'].isin(ids)

    return df.drop(df.index[shorts].tolist())

def immobs(df):
    # Group by ID
    tracks = df[['TRACK_ID', 'X', 'Y']].groupby('TRACK_ID').agg(['max', 'min'])

    # Add columns with peak-to-peak values
    tracks['diffX'] = (tracks['X']['max'] - tracks['X']['min']) <= 1.
    tracks['diffY'] = (tracks['Y']['max'] - tracks['Y']['min']) <= 1.

    # Get indexer
    cond = tracks['diffX'] & tracks['diffY']

    # Extract
    ids = tracks.index[cond].tolist()
    immobs = df['TRACK_ID'].isin(ids)

    return df.drop(df.index[immobs].tolist())

def sort(df):
    # Sort in-place
    return df.sort_values(['TRACK_ID', 'TIMESTAMP'], inplace=False)

def process(df):
    # Remove NaN entries
    df1 = nans(df)

    # Remove AV and AGENT entries
    df2 = av_agent(df1)

    # Remove entries with too short trajectories
    df3 = shorts(df2)

    # Remove entries with immobile objects
    df4 = immobs(df3)

    # Sort by ID, then by TIMESTAMP
    df5 = sort(df4)

    return df5

if __name__ == '__main__':
    # Input
    dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(dir, "input/data.csv"), index_col=False)

    # Process
    df_proc = process(df)

    # Output
    df_proc.to_csv(os.path.join(dir, "output/result.csv"), index=False)

