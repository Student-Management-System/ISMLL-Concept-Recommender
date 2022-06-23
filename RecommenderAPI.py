# Modules needed for recommendation
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import json
import os
from IPython.display import display_html
import warnings
from typing import Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F
# Moules needed for storing/loading local data
import pickle
from pathlib import Path

import json, jsonschema
from jsonschema import validate



def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace(
        'table', 'table style="display:inline"'), raw=True)


warnings.filterwarnings("ignore")


def hello_world():
    return 'Hello, World!'

def validate_json(json_data, spec):
    with open(spec, 'r') as file:
        api_schema = json.load(file)

    try:
        validate(instance=json_data, schema=api_schema)
    except jsonschema.exceptions.ValidationError as err:
        err = "Given JSON data is InValid"
        return False, err

    message = "Given JSON data is Valid"
    return True, message


        
def do_training(concept_maps_as_json, service_name):
    concepmaps = collect_data(concept_maps_as_json)
    concepmaps = concepmaps[['title', 'conceptId']] #concepmaps
    data = collect_data(concept_maps_as_json)
    data= data[['ConceptMapID','conceptId']]
    data['rating'] = 5
    data, mapping, inverse_mapping = map_column(data, col_name="conceptId")
    grp_by_train = data.groupby(by="ConceptMapID")
    random.sample(list(grp_by_train.groups), k=10)
    model = Recommender(
        vocab_size=len(mapping) + 2,
        lr=1e-4,
        dropout=0.3,
    )
    model.eval()
    concept_to_idx = {a: mapping[b] for a, b in zip(concepmaps.title.tolist(), concepmaps.conceptId.tolist()) if b in mapping}
    idx_to_concept = {v: k for k, v in concept_to_idx.items()}
    
  
    
    # Save the models
    folder = 'models/' + service_name
    Path(folder).mkdir(parents=True, exist_ok=True)
    store(model, "model", folder)
    store(concept_to_idx, "concept_to_idx", folder)
    store(idx_to_concept, "idx_to_concept", folder)

def store(data, file_name, folder):
    with open(folder + '/' + file_name, "wb") as outfile:
        # "wb" argument opens the file in binary mode
        pickle.dump(data, outfile)        
        
        
def convert_json_to_df(data):
    """This function convert JSON to DataFrame
    
       param: json_file:
       return: Final Dataframe.
    """
    
    conceptmap, users, conceptmaps_list, users_list, final_list, final_list2 = ([] for i in range(6))
    for i in data:
        d = i['conceptmap_id']
        s = i['user_id']
        conceptmap.append(d)
        users.append(s)
    my_list = ['map_id'] * len(conceptmap)
    my_list2 = ['user_id'] * len(conceptmap)
    for f, b in zip(conceptmap, my_list):
        f = [f]
        b = [b]
        intial = dict(zip(b, f))
        conceptmaps_list.append(intial) 
    for f, b in zip(users, my_list2):
        f = [f]
        b = [b]
        intial = dict(zip(b, f))
        users_list.append(intial)
    for f, b in zip(conceptmaps_list, data):
        i = b['concepts']
        result = [dict(item, **f) for item in i]
        for f in result:
            final_list.append(f) 
    for f, b in zip(users_list, data):
        i = b['concepts']
        result = [dict(item, **f) for item in i]
        for f in result:
            final_list2.append(f)
    df1 = pd.DataFrame(final_list)
    df2 = pd.DataFrame(final_list2)
    result = pd.concat([df1, df2], axis=1)
    result = result.loc[:,~result.columns.duplicated()].copy()
    result = result.drop(columns=[ 'timestamp'])
    result.rename(columns={'id': 'conceptId', 'name': 'title',  'map_id' :'ConceptMapID'}, inplace=True)
    data = result[['conceptId', 'ConceptMapID', 'title']]
    data.rename(columns={'title':'concepts'}, inplace=True)
    data['idx'] = data.groupby(['conceptId'], sort=False).ngroup()
    data = data.drop('conceptId', 1)
    data = data.rename(columns={'idx': 'conceptId'})
    return data



PAD = 0
MASK = 1


def map_column(df: pd.DataFrame, col_name: str):
    """
    Maps column values to integers
    :param df:
    :param col_name:
    :return:
    """
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    df[col_name + "_mapped"] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def get_context(df: pd.DataFrame, split: str, context_size: int = 120, val_context_size: int = 5):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows
    :param df:
    :param split:
    :param context_size:
    :param val_context_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(10, df.shape[0] - val_context_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    start_index = max(0, end_index - context_size)

    context = df[start_index:end_index]

    return context


def pad_arr(arr: np.ndarray, expected_size: int = 30):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def pad_list(list_integers, history_size: int, pad_val: int = PAD, mode="left"):
    """

    :param list_integers:
    :param history_size:
    :param pad_val:
    :param mode:
    :return:
    """

    if len(list_integers) < history_size:
        if mode == "left":
            list_integers = [pad_val] * (history_size - len(list_integers)) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (history_size - len(list_integers))

    return list_integers


def df_to_np(df, expected_size=5):
    arr = np.array(df)
    arr = pad_arr(arr, expected_size=expected_size)
    return arr



def masked_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):

    _, predicted = torch.max(y_pred, 1)

    y_true = torch.masked_select(y_true, mask)
    predicted = torch.masked_select(predicted, mask)

    acc = (y_true == predicted).double().mean()

    return acc


def masked_ce(y_pred, y_true, mask):

    loss = F.cross_entropy(y_pred, y_true, reduction="none")

    loss = loss * mask

    return loss.sum() / (mask.sum() + 1e-8)


class Recommender(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=128,
        cap=0,
        mask=1,
        dropout=0.4,
        lr=1e-4,
    ):
        super().__init__()

        self.cap = cap
        self.mask = mask

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.item_embeddings = torch.nn.Embedding(
            self.vocab_size, embedding_dim=channels
        )

        self.input_pos_embedding = torch.nn.Embedding(512, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.linear_out = Linear(channels, self.vocab_size)

        self.do = nn.Dropout(p=self.dropout)

    def encode_src(self, src_items):
        src_items = self.item_embeddings(src_items)

        batch_size, in_sequence_len = src_items.size(0), src_items.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src_items.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder)

        src_items += pos_encoder

        src = src_items.permute(1, 0, 2)

        src = self.encoder(src)

        return src.permute(1, 0, 2)

    def forward(self, src_items):

        src = self.encode_src(src_items)

        out = self.linear_out(src)

        return out

    def training_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("valid_loss", loss)
        self.log("valid_accuracy", accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        src_items, y_true = batch

        y_pred = self(src_items)

        y_pred = y_pred.view(-1, y_pred.size(2))
        y_true = y_true.view(-1)

        src_items = src_items.view(-1)
        mask = src_items == self.mask

        loss = masked_ce(y_pred=y_pred, y_true=y_true, mask=mask)
        accuracy = masked_accuracy(y_pred=y_pred, y_true=y_true, mask=mask)

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }



def train():
    if request.is_json:
        data = request.get_json()
        valid, msg = validate_json(data, 'Multiple-Concept-Maps-Recommender.spec.json')

        if not valid:
            return {"Error": "JSON request does not represent Concept Map(s):\n" + msg}, 415 # 415 means Unsupported media type

        # Do further stuff with json data
        return "Model updated", 201 # 201 means something was created
    return {"Error": "Request must be JSON"}, 415 # 415 means Unsupported media type

