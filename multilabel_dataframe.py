####################################################
#
#   Authors: Adriana Galán & Marta Goyena
#   Project: MusicTaste - Genre Classification
#   Year: 2024
#
####################################################

import pandas as pd
import csv
import glob


def process_genres(col_genres, poss_genres):
    all_genres = col_genres.split(", ")
    genres = []
    for g in all_genres:
        g3 = g.split("---")[1]
        g2 = g.split("--")[1]
        for p in poss_genres:
            if g3.lower() == p.lower() or g2.lower() == p.lower():
                genres.append(p)
    return genres


# New empty Dataframe
multilabel_data = pd.DataFrame(columns=['TRACK_ID', 'Genres'])
# Data available
datas = ['training', 'test', 'validation']
possible_genres = ['Pop', 'Alternative', 'Classical', 'Dance', 'Techno', 'Rock']

for ds in datas:
    # Paths
    files_tsv = []
    if ds == 'training':
        paths_tsv = '/Users/agv/Documents/Estudios/Universidad/MLLB/MusicTaste_project/EDA_ETL/new_data/*.tsv'
        files_tsv = glob.glob(paths_tsv)
    elif ds == 'test':
        paths_tsv = '/Users/agv/Documents/Estudios/Universidad/MLLB/test_val/*_test.tsv'
        files_tsv = glob.glob(paths_tsv)
    elif ds == 'validation':
        paths_tsv = '/Users/agv/Documents/Estudios/Universidad/MLLB/test_val/*_val.tsv'
        files_tsv = glob.glob(paths_tsv)

    # Open the tsv and extract the info
    for f in files_tsv:
        with open(f, 'r', newline='\n', encoding='utf-8') as df_tsv:
            # Crea un lector TSV
            file_tsv = csv.reader(df_tsv, delimiter='\t')

            # Itera sobre cada fila en el archivo TSV
            for row in file_tsv:
                if row == ['TRACK_ID', 'ARTIST_ID', 'ALBUM_ID', 'PATH', 'DURATION', 'TAGS']:
                    continue
                track_id = row[0].split("_")[1]
                every_genre = row[-1]

                # Process the genres
                labels = process_genres(every_genre, possible_genres)

                # Add the info to the dataframe
                multilabel_data = multilabel_data._append({'TRACK_ID': track_id, 'Genres': labels}, ignore_index=True)

    multilabel_data.to_csv(f'dataset/{ds}_multilabel.csv', sep='\t', index=False)


print(multilabel_data)
