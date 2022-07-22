# WORK IN PROGRESS

## Introduction

Proteins are improtant biomolecules comprised of amino acids joined together by peptide bonds. They are paramount to nearly all of life processes and are
its is fundamental to be able to correctly and accuractly identify proteins in a step to understand function and ecology of proteins. This has become 
especially more and more present since the arrival of covid-19 and will forever remain present as evidenced by the monekypox outbreak in may of 2022.

Machine Learning (ML) has become very relevant in humanities persuit to achieve this knowledge. A variety of ML algorithms have been applied to the problem with
varying sucess. Most notable being Transformers, and preharps LSTMs (Long short-term memory). 

This work attempts to compare these two methods. Three different points of comparision are used, all implemented in pytorch: LSTM, BERT Transformer, 
and a simple feedforward network trained on an embedding produced by ProtBert (another BERT transformer). These are all tested on the "Human vs Pathogen"
dataset curated by DeepChain.

## Methods
##### Database description

In this work, all models were assessed on a single dataset, "Human vs Pathogen" as afformentioned in the introduction. This is a human vs pathogen protein
clasification dataset. It contains 96k protein sequences (50% human, 50% pathogens) all extracted from Uniprot. The dataset also has embeddings available 
calculated with ProtBert.

The three models were trained on the "sequence" column of the dataset which contains one-dimensitonal (1D) sequences, built from twenty-four tokens (Letters),
with twenty each representing an amino acid, and the last four representing ```[mask]``` mask token, a ```[pad]``` padding token, a ```[CLS]``` classifier token,
and a ```[SEP]``` separator token. 

A couple of data pre-processing is implented before used in the model. Firstly, each token is encoded to a unqiue number so it is able to be used in the 
model and inputted into the embedding layer. This is achieved through a simple dictionary map:
```py
# Encoding tokens to a unique number
def get_seq_column_map(x):
    unique = set()
    for idx, sequence in enumerate(x[0]):
        unique.update(list(sequence))
    
    return dict(zip(unique, list(range(len(unique)))))
```
```
Output: {'M': 0, 'L': 1, 'Z': 2, 'V': 3, 'F': 4, 'S': 5, 'Y': 6, 'E': 7, 'T': 8, 'H': 9, 'U': 10, 'P': 11, 'N': 12, 'X': 13, 'R': 14, 'Q': 15, 'G': 16, 'K': 17, 'C': 18, 'W': 19, 'A': 20, 'D': 21, 'B': 22, 'I': 23}
```

Secondly, each batch must be padded to longest sequence in that batch. This is due to ```torch.utils.data.DataLoader``` requiring every sequence in the batch to 
have the same length. A simple collate function is used:
```py
def collate_padd(batch):
        x = [row[0] for row in batch]
        y = [row[1] for row in batch]
        sequence_len = [len(row) for row in x]
        
        x =  pad_sequence(x, batch_first=True)
        
        return (torch.as_tensor(x), torch.as_tensor(sequence_len)), torch.as_tensor(y)
```

Lastly, it is important to mention the data is split ~80% Training, ~10% Validation and ~10% Testing. These are arbitrary numbers and chosen for no particular reason.

##### Overview of LSTM 

The LSTM model used is very simple. It contains a ```torch.nn.Embedding``` layer to store the sequences, a ``torch.nn.LSTM`` layer which applies a
multi-layer long short-term memroy (LSTM) RNN to an input sequence, and then finally this is followed by two ```torch.nn.linear layers``` and a ```torch.nn.dropout```
layer to reach the dimesnionality of the desired target. 
```
Net(
  (embed): Embedding(24, 512)
  (lstm): LSTM(512, 256, batch_first=True)
  (linear_1): Linear(in_features=256, out_features=128, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
  (linear_2): Linear(in_features=128, out_features=1, bias=True)
)
```
***NOTE: TALK ABOUT TORCH.NN.LSTM ACTIONS***

***NOTE: TALK ABOUT SEQUENCE PACKING HERE***

##### Overview of BERT Transformer

The BERT model is like the LSTM is at its core relativley simple. It contains a ```torch.nn.Embedding``` layer to store the sequences,
```
Net(
  (embed): Embedding(24, 512)
  (pos_encoder): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0): TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=50, bias=True)
        (dropout): Dropout(p=0.5, inplace=False)
        (linear2): Linear(in_features=50, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.5, inplace=False)
        (dropout2): Dropout(p=0.5, inplace=False)
      )

      ....

      (5): TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=50, bias=True)
        (dropout): Dropout(p=0.5, inplace=False)
        (linear2): Linear(in_features=50, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.5, inplace=False)
        (dropout2): Dropout(p=0.5, inplace=False)
      )
    )
  )
  (dropout): Dropout(p=0.25, inplace=False)
  (classifier): Linear(in_features=512, out_features=1, bias=True)
)
```
***NOTE: TALK ABOUT POSTIONAL ENCODING ACTIONS***

***NOTE: TALK ABOUT TRANSFORMER ENCODER ACTIONS***


##### Overview of Classifier trained on ProtBert embedding


## Results

***NOTE: SHOW RESULTS**

## Discussion and conclusions

***NOTE: COMPARE ADVANTAGES AND DISADVANTAGES OF LSTMS AND TRANSFORMERS, ALSO EMBEDDINGS**
