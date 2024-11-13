import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class LSTMAttentionModel(nn.Module): 
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, embedding_matrix, n_layers=1, drop_prob=0.25):
      super(LSTMAttentionModel, self).__init__()
      self.batch_size = batch_size
      self.output_size = n_items
      self.input_dim = embedding_dim
      self.hidden_size = hidden_size
  
      ## Embeddings
      if embedding_matrix is not None:
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
      else: 
        self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
      self.dropout = nn.Dropout(drop_prob)

      ## RNN
      self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first = False)
      
      ## Attention
      self.query = nn.Linear(hidden_size, hidden_size)
      self.key = nn.Linear(hidden_size, hidden_size) 
      self.value = nn.Linear(hidden_size, hidden_size)
      self.softmax = nn.Softmax(dim=2) #dim=2 ##Why dimension 2?

      # Linear layer to map from hidden size to embedding size
      self.embedding_to_hidden = nn.Linear(embedding_dim, hidden_size)
      self.hidden_to_embedding = nn.Linear(hidden_size, embedding_dim)
      
    def attention_net(self, lstm_output, padding_mask): 
      queries = self.query(lstm_output)
      keys = self.key(lstm_output)
      values = self.value(lstm_output)

      score = torch.bmm(queries, keys.transpose(1, 2))/(self.hidden_size**0.5) #keys.transpose(0, 1)
      if padding_mask is not None:
        score = score.masked_fill(padding_mask.unsqueeze(1) == 0, float('-inf'))

      attention = self.softmax(score)
      weighted = torch.bmm(attention, values)
      return weighted

    def forward(self, x, lengths):
      x = x.long()
      embs = self.dropout(self.embedding(x))

      embs = pack_padded_sequence(embs, lengths)
      embs, _ = self.lstm(embs) 
      embs, lengths = pad_packed_sequence(embs)
      embs = embs.permute(1, 0, 2) # Change dimensions to: (batch_size, sequence_length, embedding_dim)

      padding_mask = (torch.sum(embs, dim=-1) != 0) 
      attn_output = self.attention_net(embs, padding_mask) 
      attn_output = torch.mean(attn_output, dim=1)  # (batch_size, hidden_size)
      attn_output = self.hidden_to_embedding(attn_output)  # Linear layer to map from hidden size to embedding size (batch_size, embedding_dim)
  
      item_embs = self.embedding(torch.arange(self.output_size).to(x.device))  
      scores = torch.matmul(attn_output, item_embs.transpose(0, 1))
      return scores