import torch
import torch.nn as nn
import torch.nn.functional as F
import math


_EPS = 1e-10




class RNNDecoder(nn.Module):
    """
    A recurrent neural network decoder for dynamic graph modeling.

    This module predicts the evolution of a graph-based system using recurrent layers with learned interactions. 
    It employs GRU-style updates to model node-level interactions and integrates edge-type-specific message-passing.

    Parameters
    ----------
    n_in_node : int
        Input feature size per node.
    edge_types : int
        Number of edge types in the graph.
    n_hid : int
        Hidden layer size.
    do_prob : float, optional
        Dropout probability (default is 0.0).
    skip_first : bool, optional
        Whether to skip the first edge type during message-passing (default is False).
    """

    def __init__(self, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('INFO:: Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):
        """
        Perform a single forward step of prediction.

        Parameters
        ----------
        inputs : torch.Tensor
            Input node features of shape [batch_size, num_nodes, n_in_node].
        rel_rec : torch.Tensor
            Receiver relation matrix of shape [num_edges, num_nodes].
        rel_send : torch.Tensor
            Sender relation matrix of shape [num_edges, num_nodes].
        rel_type : torch.Tensor
            Edge type probabilities of shape [batch_size, num_edges, edge_types].
        hidden : torch.Tensor
            Hidden states of shape [batch_size, num_nodes, n_hid].

        Returns
        -------
        pred : torch.Tensor
            Predicted node features of shape [batch_size, num_nodes, n_in_node].
        hidden : torch.Tensor
            Updated hidden states of shape [batch_size, num_nodes, n_hid].
        """
                                
        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):
        """
        Perform forward pass for multiple time steps.

        Parameters
        ----------
        data : torch.Tensor
            Input data of shape [batch_size, num_timesteps, num_nodes, n_in_node].
        rel_type : torch.Tensor
            Edge type probabilities of shape [batch_size, num_edges, edge_types].
        rel_rec : torch.Tensor
            Receiver relation matrix of shape [num_edges, num_nodes].
        rel_send : torch.Tensor
            Sender relation matrix of shape [num_edges, num_nodes].
        pred_steps : int, optional
            Prediction steps ahead (default is 1).
        burn_in : bool, optional
            Whether to use burn-in during training (default is False).
        burn_in_steps : int, optional
            Number of steps to use ground truth during burn-in (default is 1).
        dynamic_graph : bool, optional
            Whether to dynamically update the graph structure (default is False).
        encoder : nn.Module, optional
            Encoder model for dynamic graph updates.
        temp : float, optional
            Temperature for Gumbel-Softmax (default is None).

        Returns
        -------
        torch.Tensor
            Predicted trajectories of shape [batch_size, num_timesteps, num_nodes, n_in_node].
        """
                    
        inputs = data#.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = F.gumbel_softmax(logits, tau=0.5, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()

