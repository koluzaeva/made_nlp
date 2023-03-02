
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, 
                 concat_number_of_features,
                 n_tokens, 
                 n_cat_features, 
                 hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE>        
        self.conv_1 = nn.Conv1d(hid_size, 128, kernel_size=3)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.AdaptiveAvgPool1d(output_size=4)
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        # <YOUR CODE HERE>
        self.conv_2 = nn.Conv1d(hid_size, 128, kernel_size=3)
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.AdaptiveAvgPool1d(output_size=4)
        
        
        self.category_out = nn.Linear(n_cat_features, 128)# <YOUR CODE HERE>

      
        self.fc_1 = nn.Sequential(
            nn.Linear(
                in_features=concat_number_of_features,
                out_features=(hid_size // 2)
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=(hid_size // 2),
                out_features=1
            )
        )
        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.conv_1(title_beg)
        title = self.relu_1(title)
        title = self.pool_1(title)
        #print(title.size())
        # <YOUR CODE HERE>

        desc_beg = self.full_emb(input2).permute((0, 2, 1))
        desc = self.conv_2(desc_beg)
        desc = self.relu_2(desc)
        desc = self.pool_2(desc)# <YOUR CODE HERE>      
        #print(desc.size())  
        
        category = self.category_out(input3)# <YOUR CODE HERE>    
        #print(category.size())    
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            desc.view(desc.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        #print(concatenated.size())

        out = self.fc_1(concatenated)# <YOUR CODE HERE>
       
        
        return out