# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_prob=0.1):
        super(CNN, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, num_labels)

    def forward(self, input_ids):
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = input_ids
        sequence_output = self.dropout(sequence_output)
        sequence_output = sequence_output.permute(0, 2, 1) # (batch_size, hidden_size, seq_length)
        conv_output1 = self.conv1(sequence_output)
        conv_output2 = self.conv2(conv_output1)
        conv_output3 = self.conv3(conv_output2)
        # print("CONV output3", conv_output3.shape)
        # pool_output, _ = torch.max(conv_output3, dim=-1)
        conv_output3 = conv_output3.permute(0, 2, 1) # (batch_size, seq_length, hidden_size)
        # print("Pool output shape", pool_output.shape)
        logits = self.fc(conv_output3)
        return logits
