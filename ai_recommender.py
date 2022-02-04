from multiprocessing.connection import Client
import os, csv, sys, time
import argparse
import pickle

python_path = r'C:\Users\G\dev\web_ai\Scripts\python.exe'
packages = r'C:\Users\G\dev\web_ai\Lib\site-packages'

if os.path.exists(python_path):
    sys.path.append(packages)
    sys.executable = python_path
else:
    raise Exception('Executable python not found')

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

global indexed_items
indexed_items = []
global file_path
file_path = 'recommender_model.pkl'

def get_indexed_items():
    return indexed_items

def add_item_to_index(item_id):
    indexed_items.append(item_id)
    return indexed_items

def main(args):
    start_t = time.time()
    action = args.action
    if args.input:
        input_file = args.input
    if args.name:
        name = args.name
    
    if action == '1':
        path = os.path.abspath(input_file)
        if not os.path.exists(path):
            with open('readmeee.txt', 'w') as f:
                f.write('error reading file')
                f.close()
            raise Exception('File entered does not exist')

        df = pd.read_csv(path)

        indexed = get_indexed_items()
        records = []
        first_record = []
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        has_pickle = os.path.exists(file_path)

        # loop through items in csv file
        for index, row in df.iterrows():
            if index > 4:
                break
            item_id = row['id']
            item_description = row['description']

            if item_id not in indexed:

                # check if item is first id
                if item_id == 1:
                    first_record.append(item_description)
                    add_item_to_index(item_id)
                else:
                    records.append(item_description)
                    add_item_to_index(item_id)

        # ai pickle file saving and loading   
        if not has_pickle:
            # encode_first item and save pickle
            with open(file_path, "wb") as pkl:
                doc_embedding = model.encode(first_record, show_progress_bar=True)
                pickle.dump(doc_embedding, pkl)
        else:
            # load pickle file and save records
            with open(file_path, "ab") as pkl:
                doc_embedding = model.encode(records, show_progress_bar=True)
                pickle.dump(doc_embedding, pkl)

    elif action == '2':
        # check if input file exists
        path = os.path.abspath(input_file)
        if not os.path.exists(path):
            raise Exception('File entered does not exist')

        # read file and get indexed items
        df = pd.read_csv(path)
        indexed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,]

        # load ai model and pickle file
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if not os.path.exists(file_path):
            raise Exception('AI pickle file does not exists')
        with open(file_path, 'rb') as pkl:
            doc_embedding = pickle.load(pkl)
        document = torch.from_numpy(doc_embedding)

        # get item to be matched from file
        file ='all_products.csv'
        df = pd.read_csv(file)
        id = df.iloc[2].id
        query = model.encode(
            df.iloc[1].description, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )

        # perform matching

    elif action == '3':
        pass

    else:
        raise Exception('Invalid option choice')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process files for AI and Media Reformating'
    )
    parser.add_argument(
        '-a', '--action', 
        help='Action to be perfomed by script',
        required=True,
    )
    parser.add_argument(
        '-i', '--input',
        help='The file to be processed'
    )
    parser.add_argument(
        '-n', '--name',
        help='Name of object to be found'
    )
    args = parser.parse_args()
    main(args)
