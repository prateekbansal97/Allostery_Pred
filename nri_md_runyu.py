import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import time

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
class TrajectoryDataset(Dataset):
    def __init__(self, data, interval):
        self.features = data
        self.interval = interval
        self.len = data.size(0)

    def __len__(self):
        return self.interval - 1

    def __getitem__(self, idx):
        index = list(range(idx, self.len, self.interval))
        return self.features[index, :, :]

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hid, n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.0):
        super(MLPEncoder, self).__init__()
        self.mlp1 = MLP(n_in, n_hid, n_hid, dropout)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, dropout)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, dropout)
        self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, dropout)
        self.out = nn.Linear(n_hid, n_out)
        self.dropout = nn.Dropout(dropout)

    def v2e(self, x, rec, send):
        receivers = torch.matmul(rec, x)
        senders = torch.matmul(send, x)
        edge = torch.cat([receivers, senders], dim=2)
        return edge

    def e2v(self, x, rec):
        incoming = torch.matmul(rec.t(), x)
        return incoming / incoming.size(1)

    def forward(self, x, rec, send):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.mlp1(x)
        x = self.v2e(x, rec, send)
        x = self.mlp2(x)
        x_skip = x
        x = self.e2v(x, rec)
        x = self.mlp3(x)
        x = self.v2e(x, rec, send)
        x = torch.cat((x, x_skip), dim=2)
        x = self.mlp4(x)
        x = self.out(x)
        return x

class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_in_node, edge_types, n_hid, do_prob=0.):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList([nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)])

        self.msg_out_shape = n_hid

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        self.dropout_prob = do_prob

    def single_step_forward(self, x, rel_rec, rel_send, rel_type, hidden):
        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape, device=x.device)

        start_idx = 0
        norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / x.size(2)  # Average

        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(x) + self.hidden_r(agg_msgs))
        i = torch.sigmoid(self.input_i(x) + self.hidden_i(agg_msgs))
        n = torch.tanh(self.input_n(x) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = x + pred

        return pred, hidden

    def forward(self, x, rel_type, rel_rec, rel_send, pred_steps=1, burn_in=False, burn_in_steps=1):
        time_steps = x.size(1)
        hidden = torch.zeros(x.size(0), x.size(2), self.msg_out_shape, device=x.device)
        pred_all = []

        for step in range(0, x.size(1) - 1):
            if burn_in:
                if step <= burn_in_steps:
                    ins = x[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                if not step % pred_steps:
                    ins = x[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send, rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)
        return preds

class NRIMD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NRIMD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, rec, send):
        encoded = self.encoder(x, rec, send)
        edges = F.gumbel_softmax(encoded, tau=0.5, hard=True)
        prob = F.softmax(encoded, dim=-1)
        outputs = self.decoder(x, edges, rec, send, pred_steps=50, burn_in=True, burn_in_steps=49)
        return outputs, edges, prob

def one_hot(n):
    off_diag = np.ones((n, n)) - np.eye(n)
    indices = np.nonzero(off_diag)
    rec = np.eye(n, dtype=np.float32)[indices[1]]
    send = np.eye(n, dtype=np.float32)[indices[0]]
    return torch.tensor(rec).to(device), torch.tensor(send).to(device)

def train(epoch, best_val_loss):
    nll_loss_fn = nn.GaussianNLLLoss(full=False, eps=0)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)
    edges_out = []
    probs_out = []

    model.train()
    for data in train_loader:
        data = data.to(device)  # Move data to GPU
        optimizer.zero_grad()
        
        outputs, edges, prob = model(data, rec, send)

        target = data[:, 1:, :, :]

        loss_nll = nll_loss_fn(outputs, target, torch.full_like(outputs, 5e-5))
        kl_loss = kl_loss_fn(prob.log(), log_prior)
        loss = loss_nll + kl_loss
        loss.backward()
        optimizer.step()

    print(f'Training epoch: {epoch}, Loss: {loss.item()}')
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  # Move data to GPU
            outputs, edges, prob = model(data, rec, send)
            target = data[:, 1:, :, :]

            loss_nll = nll_loss_fn(outputs, target, torch.full_like(outputs, 5e-5))
            kl_loss = kl_loss_fn(prob.log(), log_prior)
            loss = loss_nll + kl_loss

            val_loss += loss

        print(f'Validation epoch: {epoch}, Loss: {val_loss.item()}')

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    return best_val_loss
# Hyperparameters and Setup
torch.manual_seed(0)
data_array = np.load("data.npy")
models, num_residues, dims = data_array.shape
features = torch.from_numpy(data_array).to(device)
interval = 100
timesteps = (models + 1) // interval
batch_size = 1
num_hidden = 6
edge_types = 2
dataset = TrajectoryDataset(features, interval)
train_len = int(len(dataset) * 0.8)
val_len = len(dataset) - train_len - 1

train_data, val_data, test_data = random_split(dataset, [train_len, val_len, 1])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
rec, send = one_hot(num_residues)
log_prior = torch.FloatTensor(np.log([0.2, 0.8])).view(1, 1, -1).expand(1, (num_residues - 1) * num_residues, -1).to(device)

encoder = MLPEncoder(timesteps * dims, num_hidden, edge_types, dropout=0.3).to(device)
decoder = RNNDecoder(dims, edge_types, num_hidden, do_prob=1e-8).to(device)
model = NRIMD(encoder, decoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
best_val_loss = float('inf')
for epoch in range(50):
    best_val_loss = train(epoch, best_val_loss)

model.eval()
for data in test_loader:
    with torch.no_grad():
        _, _, probs = model(data, rec, send)
        output_np = probs.cpu().numpy()
np.save("output_matrix.npy", output_np)
