
import torch
import torch.nn as nn

class MultiOutputLSTMWithEmb(nn.Module):
    def __init__(self, input_dim, emb_num, emb_dim=8, hidden_dim=64, num_layers=2, output_dim=7, dropout_p=0.3):
        super().__init__()
        self.menu_emb = nn.Embedding(emb_num, emb_dim)
        self.lstm = nn.LSTM(input_dim + emb_dim, hidden_dim, num_layers, batch_first=True)

        self.res_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.res_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, menu_id):
        emb = self.menu_emb(menu_id)
        emb_expanded = emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x_cat = torch.cat([x, emb_expanded], dim=-1)
        out, _ = self.lstm(x_cat)
        last_hidden = out[:, -1, :]
        residual = last_hidden
        x_res = self.res_fc1(last_hidden)
        x_res = self.relu(x_res)
        x_res = self.dropout(x_res)
        x_res = self.res_fc2(x_res)
        x_res = x_res + residual
        return self.fc_out(x_res)
