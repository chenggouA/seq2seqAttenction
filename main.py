import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.load import tokenizes, get_train_test
from model.seq2seq import seq2seq, Decoder, Encoder
from torch import nn

batch_size = 8

X_train, X_test, y_train, y_test = get_train_test(0.5)

X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建 Pytorch Dataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# ======== model create
vocab_len = len(tokenizes)
emb_dim = 300
hidden = 32
n_layers = 3
device = 'cuda'


encoder_infer = Encoder(vocab_len, emb_dim, hidden, n_layers, 0.5)
encoder_infer.to(device)
decoder_infer = Decoder(vocab_len, emb_dim, hidden, n_layers, 0.5)
decoder_infer.to(device)
model = seq2seq(encoder_infer, decoder_infer)
# 交叉熵损失
criterion = nn.CrossEntropyLoss()

#  SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.to(device)

EPOCH = 4
# ======== 

for i in range(EPOCH):
    print(f"============Epoch: {i + 1}============")
    loss = 0
    for X_batch, y_batch in train_dataloader:
        X_batch = X_batch.to(device)
        
        y_batch = y_batch.to(device)
        output = model(X_batch, y_batch)
        seq_len = output.shape[0]
        # 清空X_batch
        del X_batch
        # 计算每个时间步的损失

        for i in range(1, seq_len):
            # 略过了第一个时间步
            loss += criterion(output[i], y_batch[:, i]) 

        del y_batch
        del output

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print(f"loss: {loss: .6f}")

    

    
