import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tflearn.data_utils import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from sklearn.metrics import classification_report
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer

#超参数和全局变量
parser = argparse.ArgumentParser("DGA")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
# parser.add_argument('--layers', type=int, default=20, help='total number of layers')
args = parser.parse_args()

#最大域名的长度，预处理时用到
max_document_length=38
#特征的个数
#embedding_dim=30
#记录分类的类别数
classfication=2
#记录卷积核数量，也就是第二维的维度
out_channel=64



#定义设备,把八个GPU都用上
if(torch.cuda.is_available()):
    device=torch.device("cuda:0")
else:
    device=torch.device("cpu")
alexa1000000_path = "E:\\论文\\冷宝\\2017-11-16-dgarchive\\alexa_1m.csv"

class Dataset(Dataset):
	def __init__(self, x, label):
		super(Dataset, self).__init__()
		self.x = x
		self.label = label

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		return self.x[index], self.label[index]

class Reines(nn.Module):
    def __init__(self):
        super(Reines,self).__init__()
        # self.embedding1=nn.Embedding(128,embedding_dim,padding_idx=0)   #前一个指编码中的最大数，后一个指第三维是多少
        #卷积层，第一个是卷积核，然后最大池化，然后
        self.RobertaModel = RobertaModel.from_pretrained("E:\\论文\\冷宝\\Reproduct-latest-DGA-tasks-main\\Code\\model\\roberta-base")
        self.Cnn_model2=nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=out_channel,kernel_size=2,stride=1),#我这里还是第二维是域名的长度，第三维是特征，out_channel理解为卷积核的个
            nn.MaxPool1d(kernel_size=2)  #这个地方就是第三个维度直接用输入除以输出
        )
        self.Cnn_model3=nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=out_channel,kernel_size=3,stride=1),#我这里还是第二维是域名的长度，第三维是特征，out_channel理解为卷积核的个
            nn.MaxPool1d(kernel_size=2)  #这个地方就是第三个维度直接用输入除以输出
        )
        self.Cnn_model4=nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=out_channel,kernel_size=4,stride=1),#我这里还是第二维是域名的长度，第三维是特征，out_channel理解为卷积核的个
            nn.MaxPool1d(kernel_size=2)  #这个地方就是第三个维度直接用输入除以输出
        )
        self.Cnn_model5=nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=out_channel,kernel_size=5,stride=1),#我这里还是第二维是域名的长度，第三维是特征，out_channel理解为卷积核的个
            nn.MaxPool1d(kernel_size=2)  #这个地方就是第三个维度直接用输入除以输出
        )
        self.lstm1=nn.LSTM(input_size=768,hidden_size=128,num_layers=1,batch_first=True,bidirectional=True)  #batch_first表示是否将batch放在第一个,hidden_size表示多少个神经元
        self.multiheadattention=nn.MultiheadAttention(embed_dim=256,num_heads=4,batch_first=True)


        self.linear1=nn.Linear(256,256)
        self.linear2=nn.Linear(256,256)
        self.linear3=nn.Linear(256,256)

        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)
        self.linear=nn.Linear(5248, classfication)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # x = self.embedding1(x)  # 维度是(batch_size,seq_length,embedding_dim) (64,48,30)
        # x= x.permute(0, 2, 1)  # (64,30,48)
        x = self.RobertaModel(input_ids=input_ids, attention_mask=attention_mask)
        print(x)
        output, (h_n, c_n) = self.lstm1(x[0])  # 维度是(batch_size,seq_length,hidden_size*2) (64,38,256)
        tmp1 = self.linear1(output)    # (64,38,256)
        tmp2 = self.linear2(output)    # (64,38,256)
        tmp3 = self.linear3(output)    # (64,38,256)
        attn_output, attn_output_weights = self.multiheadattention(tmp1, tmp2, tmp3)  # (query,key,value) (64,38,256)  (64,38,38)
        attn_output = attn_output.permute(0, 2, 1)
        # print(attn_output.shape, attn_output_weights.shape)
        # x1 = self.Cnn_model2(x)   # 按in_channel=48时，输出的维度为(64,48,14) 但是网上看in_channel应该为词向量维度所以应该为30？
        # x2 = self.Cnn_model3(x)
        # x3 = self.Cnn_model4(x)
        # x4 = self.Cnn_model5(x)
        x1 = self.Cnn_model2(attn_output) #(64,256,18)
        x2 = self.Cnn_model3(attn_output) #(64,256,18)
        x3 = self.Cnn_model4(attn_output) #(64,256,17)
        x4 = self.Cnn_model5(attn_output) #(64,256,17)
        x5 = torch.cat((x1, x2, x3, x4), dim=2)  # 这个地方是最终的输入维度（batch_size,out_channel,70）
        output = self.flatten(x5)  # 维度是(batch_size,max_document_length*hidden_size) (64,4480)
        output = torch.cat((x[1], output), dim=-1)


        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output

def load_alexa():
    x=[]
    data = pd.read_csv(alexa1000000_path, header=None)
    x=[i[0] for i in data.values]   #读alexa10000_path的这个txt并把第一列的数据存入x列表中
    return x
def load_dga(path):
    x=[]
    data = pd.read_csv(path, header=None)
    x=[i[0] for i in data.values]
    return x

def get_feature_charseq():
    alexa = load_alexa()
    x = alexa
    y = []
    for i in range(0, 100000):
        y.append(0)
    y = torch.Tensor(y)
    #print(y)
    # t = []
    # for i in x:  # 先将字符转ASCII值，把所有域名转换为数字，这时没有word embedding也就没有学习字符之间的关系，只是将他们转成了一个个数字。
    #     v = []
    #     for j in range(0, len(i)):
    #         v.append(ord(i[j]))
    #     t.append(v)
    #
    # x = t
    #
    # x = pad_sequences(x, maxlen=max_document_length, value=0.)  # 数据预处理
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)
    return x_train, x_test, y_train, y_test


if __name__=='__main__':
    x_train, x_test, y_train, y_test = get_feature_charseq()
    train_data = Dataset(x=x_train, label=y_train)
    test_data = Dataset(x=x_test, label=y_test)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=True)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=6, drop_last=True)  #默认batchsize为64，为一个batch
    # for i in train_queue:
    #     print(len(i[0]))
    model = Reines()
    model = model.to(device)
    # print(model)
    tokenizer = RobertaTokenizer.from_pretrained("E:\\论文\\冷宝\\Reproduct-latest-DGA-tasks-main\\Code\\model\\roberta-base")
    model1 = RobertaModel.from_pretrained("E:\\论文\\冷宝\\Reproduct-latest-DGA-tasks-main\\Code\\model\\roberta-base").to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 开始训练
    for i in range(args.epochs):
        # 记录训练的次数
        total_trainstep = 0
        # 记录测试的次数
        total_teststep = 0
        print("--------------第{}轮训练开始--------------".format(i + 1))

        model.train()
        for data in train_queue:
            input, label=data
            text1 = "equipauto-algeria.net"
            encode_input = tokenizer(text1, return_tensors='pt', add_special_tokens=False, max_length=max_document_length, padding='max_length').to(device)
            output = model1(**encode_input)
            print(output)
            input_ids = Variable(encode_input['input_ids'])
            attention_mask = Variable(encode_input['attention_mask'])
            label = Variable(label)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label=label.to(device)
            # output = model1(input_ids=input_ids, attention_mask=attention_mask)
            # print(output)
            output = model(input_ids, attention_mask)
            # print(output)
            # print(output.argmax(1))
            loss=loss_fn(output, label.long())
            # print(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_trainstep = total_trainstep + 1

            if (total_trainstep % 100 == 0):
                print("训练次数：{} , loss :{}".format(total_trainstep, loss))
        # 测试步骤
        model.eval()
        total_test_loss = 0
        # total_accuracy=0
        # sum=0
        final_true = []
        final_pridict = []
        with torch.no_grad():
            for data1 in test_queue:
                input1, label1 = data1
                input1 = Variable(input1)
                label1 = Variable(label1)
                input1 = input1.to(device)
                label1 = label1.to(device)

                output1 = model(input1)
                loss1 = loss_fn(output1, label1.long())
                # print(output1)
                # print(output1.argmax(1))
                # print(label1)
                for j in range(0, args.batch_size):
                    final_pridict.append(output1.cpu().argmax(1)[j])   # .cpu是将数据类似从cuda转到cpu上，不改变数据类型，转换后仍是tensor
                    final_true.append(label1.cpu()[j])

                total_teststep = total_teststep + 1

                if (total_teststep % 100 == 0):
                    print("测试次数：{} , loss :{}".format(total_teststep, loss1))
                # print(output.argmax(1))
                # print(label)
                # sum=sum+1
                # accuracy=(output1.argmax(1)==label1).sum()
                total_test_loss = loss1 + total_test_loss
                # total_accuracy=total_accuracy+accuracy
        print("整体测试集上的loss为{}".format(total_test_loss))
        # print("整体测试集上的准确率为:{}".format(total_accuracy/(sum*args.batch_size)))
        print(classification_report(final_true, final_pridict))

