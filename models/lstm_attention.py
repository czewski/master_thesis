import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTMAttentionModel(nn.Module): 
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers=1, drop_prob=0.25, num_heads=4):
        super(LSTMAttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = n_items
        self.input_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

        # Embeddings
        self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(drop_prob)

        # RNN
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=False)

        # Multi-Head Attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

        # Linear layer to map from hidden size to embedding size
        self.embedding_to_hidden = nn.Linear(embedding_dim, hidden_size)
        self.hidden_to_embedding = nn.Linear(hidden_size, embedding_dim)
        self.output_projection = nn.Linear(hidden_size, hidden_size)  # Project back after multi-head

    def find_closest_tensor(self, query_embeddings):
      closest_tensors = []
      
      for t in range(query_embeddings.size(0)):  # iterate over sequence length
          # query: (batch_size, dimension) at each time step t
          query_at_t = query_embeddings[t]
          data_embeddings = self.embedding.weight.data
          
          # Calculate cosine similarity for each batch 
          dists = 1 - F.cosine_similarity(query_at_t.unsqueeze(1), data_embeddings.unsqueeze(0), dim=-1)
          min_dist, closest_index = torch.min(dists, dim=1) 
          closest_tensor = data_embeddings[closest_index] 
          
          closest_tensors.append(closest_tensor) 
      
      return torch.stack(closest_tensors)

    def attention_net(self, lstm_output, padding_mask): 
        batch_size, seq_len, hidden_size = lstm_output.size()

        # Linear projections for multi-head attention
        queries = self.query(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled Dot-Product Attention
        score = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if padding_mask is not None:
            score = score.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attention = self.softmax(score)  # (batch_size, num_heads, seq_len, seq_len)
        weighted = torch.matmul(attention, values)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        weighted = weighted.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        multi_head_output = self.output_projection(weighted)  # Linear projection after concatenation
        return multi_head_output

    def forward(self, x, lengths):
        x = x.long()
        embs = self.dropout(self.embedding(x))

        closest_tensor = self.find_closest_tensor(self.embedding(x))  # Use KNN to find the closest tensor in the dataset  
        #closest_tensor = torch.mean(closest_tensor, dim=0) 
        #print(closest_tensor.size())
        embs = embs * closest_tensor

        # Pack sequence
        embs = pack_padded_sequence(embs, lengths)
        embs, _ = self.lstm(embs)
        embs, _ = pad_packed_sequence(embs)
        embs = embs.permute(1, 0, 2)  # Change dimensions to: (batch_size, sequence_length, hidden_size)

        padding_mask = (torch.sum(embs, dim=-1) != 0)
        attn_output = self.attention_net(embs, padding_mask)
        attn_output = torch.mean(attn_output, dim=1)  # (batch_size, hidden_size)
        attn_output = self.hidden_to_embedding(attn_output)  # Map to embedding size (batch_size, embedding_dim)

        # Dot product with item embeddings
        item_embs = self.embedding(torch.arange(self.output_size).to(x.device))
        scores = torch.matmul(attn_output, item_embs.transpose(0, 1))
        return scores
