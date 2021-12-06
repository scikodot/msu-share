import os
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dir_ = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(dir_, "input/data.csv"), index_col=False)

    # Get legend labels
    df['LABEL'] = df['TRACK_ID'].str.split('-').str[-1] + ' (' + df['OBJECT_TYPE'] + ')'

    fig, ax = plt.subplots(1, figsize=(16, 16))
    markers = {'AV': 'd', 'AGENT': 'x', 'OTHERS': 'o'}

    for obj in df['TRACK_ID'].unique():
        data = df[df['TRACK_ID'] == obj][['OBJECT_TYPE', 'X', 'Y']]
        obj_type = data['OBJECT_TYPE'].iloc[0]
        ax.plot(data['X'], data['Y'], marker=markers[obj_type])

    ax.grid()
    ax.set_title('Визуализация дорожной сцены', fontsize=20)
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.legend(df['LABEL'].unique(), loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.savefig(os.path.join(dir_, "output/scene.png"), dpi=300, bbox_inches='tight')
