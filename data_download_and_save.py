"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506171442
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from metapub import PubMedFetcher


df = pd.read_excel(
    './CDC_datasetML.xlsx', 
    sheet_name='trainset_2286', 
)
df.head()


fetch = PubMedFetcher()
error_list = []
art_dataset = []
for idx in tqdm(range(len(df))):
    pmid = df.iloc[idx]['PMID']
    # print(str(pmid) + ' ', end='')
    try:
        art = fetch.article_by_pmid(pmid)
        if art.title is None:
            error_list += [{'pmid': int(pmid), 'art': 'title_None'}]
            continue
        if art.abstract is None:
            error_list += [{'pmid': int(pmid), 'art': 'abstract_None'}]
            continue
        # if art.authors is None:
        #     error_list += [{'pmid': int(pmid), 'art': 'authors_None'}]
        #     continue
        # if art.journal is None:
        #     error_list += [{'pmid': int(pmid), 'art': 'journal_None'}]
        #     continue

        art_dataset += [{
            'pmid': int(pmid), 
            'title': art.title, 
            'abstract': art.abstract, 
            # 'authors': [str(author) for author in art.authors],
            # 'journal': art.journal,
            'label': int(df.iloc[idx]['Curate (0: T0, 1: T2/4)']),
            'split': 'train'
        }]

    except Exception as e:
        error_list += [{'pmid': int(pmid), 'art': e}]

print(len(art_dataset))
print(error_list)



df = pd.read_excel(
    './CDC_datasetML.xlsx', 
    sheet_name='testset_400', 
)
df.head()


for idx in tqdm(range(len(df))):
    pmid = df.iloc[idx]['PMID']
    # print(str(pmid) + ' ', end='')
    try:
        art = fetch.article_by_pmid(pmid)
        if art.title is None:
            error_list += [{'pmid': int(pmid), 'art': 'title_None'}]
            continue
        if art.abstract is None:
            error_list += [{'pmid': int(pmid), 'art': 'abstract_None'}]
            continue
        # if art.authors is None:
        #     error_list += [{'pmid': int(pmid), 'art': 'authors_None'}]
        #     continue
        # if art.journal is None:
        #     error_list += [{'pmid': int(pmid), 'art': 'journal_None'}]
        #     continue

        art_dataset += [{
            'pmid': int(pmid), 
            'title': art.title, 
            'abstract': art.abstract, 
            # 'authors': [str(author) for author in art.authors],
            # 'journal': art.journal,
            'label': int(df.iloc[idx]['Curate (0: T0, 1: T2/4)']),
            'split': 'valid'
        }]

    except Exception as e:
        error_list += [{'pmid': int(pmid), 'art': e}]

print(len(art_dataset))
print(error_list)

def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

file_save_path = './dataset/data_2506101500.json'
with open(file_save_path, "w") as file:
    json.dump(art_dataset, file, indent=4, default=convert)











