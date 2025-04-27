import torch.nn as nn
from .fusion import Bottleneck
import  torch.nn.functional
class MLP_block(nn.Module):

    def __init__(self, feature_dim, output_dim, dropout, attention_config):
        super(MLP_block, self).__init__()
        self.feature_dim = feature_dim
        self.activation = nn.ReLU(inplace=True)

        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128) #BatchNorm1d
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.layer4 = nn.Linear(64, output_dim)
        self.drop = nn.Dropout(0.1)
        self.attention = Bottleneck(inplanes=attention_config['INPUT_DIM'], 
                                    planes=attention_config['HIDDEN_DIM'],
                                    base_width=attention_config['BASE_WIDTH'],
                                    fuse_type=attention_config['FUSE_TYPE'])

    def forward(self, x):
        B, C, H, W = x.shape

        assert self.feature_dim == H*W, \
            f"Argument --INPUT_FEATURE_DIM in config['MODEL']['EVALUATOR'] should be equal to {H*W} (num_modal x feature_dim of each branch))"

        x = self.attention(x).view(B, -1)

        x = self.activation(self.bn1(self.layer1(x)))
        x= self.drop(x)
        x = self.activation(self.bn2(self.layer2(x)))
        x= self.drop(x)
        x = self.activation(self.bn3(self.layer3(x)))
        x = self.drop(x)
        output = self.softmax(self.layer4(x))

        return output


class Evaluator(nn.Module):

    def __init__(self, feature_dim, output_dim,  dropout, attention_config):
        super(Evaluator, self).__init__()
        self.evaluator = MLP_block(feature_dim, output_dim, dropout, attention_config)


    def forward(self, feats_avg):
        probs = self.evaluator(feats_avg)
        
        return probs


