import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num  #标签个数
        chanel_num = 1
        filter_num = args.filter_num  #卷积核个数
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        #如果使用预训练词向量，提前加载，不需要微调时设置freeze为True
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
            
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, class_num)

    def forward(self, x):
        # 经过embedding
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        # 经过卷积运算
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # 经过最大池化层
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        # 将不同卷积核提取的特征组合起来
        x = torch.cat(x, 1)
         # dropout层
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
