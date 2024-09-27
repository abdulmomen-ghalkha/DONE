import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 1)(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        output = x#F.log_softmax(x, dim=1)
        return output
        
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted input size based on the dimensions after pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # Adjusted input size based on the dimensions after pooling
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv2(nn.ReLU()(self.conv1(x))))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv4(nn.ReLU()(self.conv3(x))))
        x = self.pool2(x)
        x = nn.ReLU()(self.conv6(nn.ReLU()(self.conv5(x))))
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 8 * 8, 64)  # Adjusted input size based on the dimensions after pooling
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        x = nn.ReLU()(self.conv3(x))

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class DNN(nn.Module):
    def __init__(self, input_dim = 784, mid_dim = 100, output_dim = 10):
        super(DNN, self).__init__()
        # define network layers
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        # define forward pass
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.log_softmax(x, dim=1)
        return x

class Linear_Regression(nn.Module):
    def __init__(self, input_dim = 40, output_dim = 1):
        super(Linear_Regression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

class Logistic_Regression(nn.Module):
    def __init__(self, input_dim = 40):
        super(Logistic_Regression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        outputs = torch.sigmoid(outputs)
        return outputs



class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

class CharLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        n_hidden,
        num_embeddings,
        embedding_dim,
        max_seq_len,
        dropout_rate,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_hidden = n_hidden
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.n_hidden,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        seq_lens = torch.sum(x != (self.num_embeddings - 1), 1) - 1
        x = self.embedding(x)  # [B, S] -> [B, S, E]
        out, _ = self.lstm(x)  # [B, S, E] -> [B, S, H]
        out = out[torch.arange(out.size(0)), seq_lens]
        out = self.fc(self.dropout(out))  # [B, S, H] -> # [B, S, C]
        return out



def Sent():
    model = CharLSTM(
    num_classes=2,
    n_hidden=100,
    num_embeddings=101,
    embedding_dim=100,
    max_seq_len=25,
    dropout_rate=0.1,
)
    return model