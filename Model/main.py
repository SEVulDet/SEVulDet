from sklearn.ensemble import RandomForestClassifier as RFC
from vectorize_gadget import GadgetVectorizer
from clean_gadget import clean_gadget
from sklearn.externals import joblib
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import *
from prepare_data import *
from cnn import CnnNet
import numpy as np
import torch


# hyparameters
seed = 1
lr = 0.0001
epoches = 20
batch_size = 16
output_size = 1

hidden_size = 256
num_layers = 2
embed_dim = 30
bidirectional = True
dropout = 0.5
use_cuda = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

save_filename = "default_model"

if __name__ == "__main__":
    base = input("input filepath:")
    sf = input("output filename:")
    if sf != "":
        save_filename = sf
    x, y, inds = getDataLoader(base + '.txt')
    print("Data loaded!")
    mydataset = MyDataset(x, y, inds)
    train_size = int(0.8 * mydataset.__len__())
    test_size = mydataset.__len__() - train_size
    print(f'train_size:{train_size}, test_size:{test_size}')
    train_dataset, test_dataset = torch.utils.data.random_split(mydataset, [train_size, test_size])
    train_loader, test_loader = Data.DataLoader(train_dataset, 1, shuffle=True, num_workers=0, drop_last=True,
                                                collate_fn=collate_fn), Data.DataLoader(test_dataset, 1,
                                                                                        shuffle=True,
                                                                                        num_workers=0,
                                                                                        drop_last=True,
                                                                                        collate_fn=collate_fn)
    cnn_net = CnnNet(in_channels=2,
                     out_channels=256,
                     kernel_width=9,
                     embed_dim=embed_dim)


    if use_cuda:
        cnn_net = cnn_net.cuda()
    optimizer = torch.optim.Adamax(cnn_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, total_step = 0, 0
    for epoch in range(epoches):
        for step, (batch_x, batch_y, index) in enumerate(train_loader):
            total_step += 1
            if step == 0:
                optimizer.zero_grad()
            batch_x, batch_y = Variable(batch_x).cuda().float(), Variable(batch_y).cuda().float().view(1, -1)
            batch_x = batch_x.reshape(1, -1, embed_dim)
            target = cnn_net(batch_x)  
            cur_loss = criterion(target, batch_y)
            cur_loss.backward()
            total_loss += cur_loss.item()
            if step % batch_size - 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
            elif step % 500 == 499:
                log('Epoch:{}, Step:{}, Aver_loss:{:2.8f}, Cur_loss:{:2.8f}'.format(epoch, step,
                                                                                    total_loss / total_step,
                                                                                    cur_loss.item()))
        evaluate()
        torch.save(cnn_net.state_dict(), 'result/' + save_filename + '_' + str(epoch) + '.pkl')
    print("Last Evaluate:")
    evaluate(True)
