## Introduction

Proteins are important biomolecules comprised of amino acids joined together by peptide bonds. They are paramount to nearly all of life's processes and  it is fundamental to be able to correctly and accurately identify proteins in a step to understand function and ecology of them. This has become especially more and more present since the arrival of COVID-19 and will forever remain present as evidenced by the monkeypox outbreak in May 2022.

Machine Learning (ML) has become very relevant in humanity's pursuit to achieve this knowledge. A variety of ML algorithms have been applied to the problem with varying success. Most notable being Transformers, and perhaps LSTMs (Long short-term memory). 

This work attempts to compare these two methods. Three different points of comparison are used, all implemented in PyTorch: LSTM, BERT Transformer, and a simple feedforward network trained on an embedding produced by ProtBert (another BERT transformer). These are all tested on the "Human vs Pathogen" dataset curated by DeepChain.

## Methods
##### Database description

In this work, all models were assessed on a single dataset, "Human vs Pathogen" as aforementioned in the introduction. This is a human vs pathogen protein classification dataset. It contains 96k protein sequences (50% human, 50% pathogens) all extracted from Uniprot. The dataset also has embeddings available calculated with ProtBert.

The three models were trained on the "sequence" column of the dataset which contains one-dimensional (1D) sequences, built from twenty-four tokens (Letters), with twenty each representing an amino acid, and the last four representing a ```[mask]```  mask token, a ```[pad]``` padding token, a ```[CLS]``` classifier token, and a ```[SEP]``` separator token. 

A couple of data pre-processing is implemented before being used in the model. Firstly, each token is encoded to a unique number so it is able to be used in the model and inputted into the embedding layer. This is achieved through a simple dictionary map:
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

Secondly, each batch must be padded to the longest sequence in that batch. This is due to ```torch.utils.data.DataLoader``` requiring every sequence in the batch to have the same length. A simple collate function is used:
```py
def collate_padd(batch):
        x = [row[0] for row in batch]
        y = [row[1] for row in batch]
        sequence_len = [len(row) for row in x]
        
        x =  pad_sequence(x, batch_first=True)
        
        return (torch.as_tensor(x), torch.as_tensor(sequence_len)), torch.as_tensor(y)
```

Lastly, it is important to mention the data is split ~80% Training, ~10% Validation and ~10% Testing. These are arbitrary numbers and were chosen for no particular reason.

##### Overview of LSTM 

The LSTM model used is very simple. It contains a ```torch.nn.Embedding``` layer to store the sequences, a ``torch.nn.LSTM`` layer which applies a multi-layer long short-term memory (LSTM) RNN to an input sequence, and then finally this is followed by two ```torch.nn.linear layers``` and a ```torch.nn.dropout``` layer to reach the dimensionality of the desired target. 
```
Net(
  (embed): Embedding(24, 512)
  (lstm): LSTM(512, 256, batch_first=True)
  (linear_1): Linear(in_features=256, out_features=128, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
  (linear_2): Linear(in_features=128, out_features=1, bias=True)
)
```
The PyTorch ```torch.nn.LSTM``` is the main layer in the model. For each element in the input sequence, each layer computes the following function. 

![LSTM function equations](https://user-images.githubusercontent.com/71031687/112730543-73de0900-8f32-11eb-8396-a79091979335.JPG)

Where h<sub>t</sub> is the hidden state at time t, c<sub>t</sub> is the cell state at time t, X<sub>t</sub> is the input at time t, h<sub>t-1</sub> is the hidden state of the layer at time <sub>t-1</sub> or the initial hidden state at time <sub>o</sub>, and i<sub>t</sub>, f<sub>t</sub>, o<sub>t</sub> are the input, cell, and output gates, respectively. The equations and the architecture of a single LSTM cell can be visualised well with the following diagram:

![LSTM architecture](https://miro.medium.com/max/1400/1*ahafyNt0Ph_J6Ed9_2hvdg.png)

It is also important to note that before the sequence is put into the LSTM, it is packed. This is done because of the large variation between the length of input sequences resulting in certain sequences having large amounts of padding. Packing the sequence reduces the amount of computing required significantly since we remove all the pad tokens and therefore no longer have to process them.

```py
packed_input = pack_padded_sequence(embed, sequence_len, batch_first=True, enforce_sorted=False)
lstm_1_seq, _ = self.lstm(packed_input)
output, _ = pad_packed_sequence(lstm_1_seq, batch_first=True)
```

##### Overview of BERT Transformer

The BERT model is like the LSTM is at its core relatively simple. It contains a ```torch.nn.Embedding``` layer to store the sequences, before a ```PositionalEncoding``` layer to describe location which is followed by 6 layers of ```torch.nn.TransformerEncoderLayer```. There is no decoder since this model is a BERT Transformer. Lastly, this is all followed a fully connected layer.
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
A vital part of the BERT transformer is the positional encoding layer. It is added to the model before the encoder explicity to retain information regarding the order of amino acids in a sequence. It works by describing the location of a token in a sequence so that each position is assigned a unique representation. Each position is mapped to a vector and therefore the ouput of the positional encoding layer is a matrix, where each row represents an encoded object of the sequence.

The Positional encoder simply follows a relatively simple equation.

![Positional Encoding Equation](https://i.stack.imgur.com/67ADh.png)

This can be implemented in a Positonal Encoding module:
```py
# Taken directly from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

After ```PositionalEncoding``` module injects some information about the position of the tokens in the sequence, it is followed by six encoder layers. Each layer has 8 heads, performing multi-headed attention. This means the attention module repeats its computations multiple times in parellel. The attention module achieves this by spliting its Query, Key and Value parameters N-ways and passes each split independently through each head. This is all done with the ```torch.nn.TransformerEncoder```.

```py
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)
self.transformer_encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=num_layers,
)
```

The entire multi-headed attetnion section of the model can be visualised well using this diagmram:
![Transformer Multi-Headed Attention](https://production-media.paperswithcode.com/methods/multi-head-attention_l1A3G7a.png)

##### Overview of Classifier trained on ProtBert embedding

The classifier trained on ProtBert embedding is a very interesting model since it shows the power of transfer learning. The model is incredibly basic, only consisting of two linear layers, with an activation function in between, and a softmax function at the end.

```
Net(
  (linear1): Linear(in_features=1024, out_features=2, bias=True)
  (activation): ReLU()
  (linear2): Linear(in_features=2, out_features=1, bias=True)
  (softmax): Softmax(dim=1)
)
```
The model is trained on an embedding rather than the data defined earlier (and used to train the LSTM and BERT Transformer). This embedding is calculated using ProtBert model. The protbert model is a more refined and complex version of the BERT Transformer used in this work. The emebdding is 1024 with a dimension of thousand and twentyfour, hence the ```in_features=1024``` in the input layer.

## Results

The results between each different model varied. Each model had a different accuracy and training time despite using the same data.

| Model              | Accuracy      | Train time / epoch / seconds |
| -------------------|:-------------:| ----------------------------:|
| LSTM               |         0.897 |                          <10 |
| BERT Transformer   |         0.    |                              |
| ProtBert Embedding |         0.949 |                         >400 |


## Discussion and conclusions

The different approaches to solving this challenging issue show the power of transformers and the clear drawbacks of LSTMs (and all RNNs). This can be noted in both the speed and the accuracy of the models, especially for this task which requires the classification of sometimes very long sequences.

The LSTM proved to be the least effective in classifying the sequence, this is due to the well-known drawbacks of LSTMs. Firstly, LSTMs, although solving the problem of vanishing gradients partially, fail to remove the problem completely. The data still has to move from cell to cell for its evealutation. In addition to this, LSTMs are vulnerable to overfitting although this problem is dampened by the dropout layers. In addition to this, the LSTM proved to be the most time-consuming to train, this is obviously from the sequential computation in the LSTM layer, as the LSTM has to calculate the hidden layers iteratively, having to weight for the hidden state at time t-1 to calculate the hidden state at time t.

The BERT Transformer

The basic classifier trained on ProtBert embedding was unsuprisingly by far the most effective. ProtBert's model is trained on a far larger dataset which gives it some advantage, however, more importantly, the actual model itself is well designed for this classification task designed by RostLab. The true power of transfer learning is on display in this model. With it scoring the highest accuracy and fastest training time.
