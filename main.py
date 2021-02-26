# This is a sample Python script.
#
## Analizzare la board e dire chi sta vincendo basandosi sulla probabilita
#

import numpy as np
# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import os

# 1 colonna posizione sulla scacchiera 1 colonna punteggio


batch_size = 100
lr = 0.01


# convert fen string to binary array (THIS IS THE GOOD FUNCTION)
def fen2matrix(fen):
    # if black active we need to reverse the checkboard
    # so black = white with reversed checkboard (in this way we achieve symmetric evaluation)
    # The bad point is that black and white games are not symmetrical, especially in opening
    # Also for sake of simplicity we ignore the castle marker information (not so relevant)

    fen_spl = fen.split()
    piece_plac = fen_spl[0] #pieces placement
    is_white = fen[1]=="w" #get color active
    matrix_pos = np.zeros([12, 8, 8]).astype('float32')

    #if black active we reverse the pieces order and the score
    if is_white:
        pieces = list("RNBQKPrnbqkp")
    else:
        pieces = list("rnbqkpRNBQKP")
    col = 0
    row = 0
    for c in piece_plac:
        for i, p in enumerate(pieces):
            if c == p:
                #if black is active we reverse rows order
                if is_white:
                    matrix_pos[i, row, col] = 1
                else:
                    matrix_pos[i, 7-row, col] = 1
                col += 1
                break
        if c == '/':
            row += 1
            col = 0
        elif c.isdigit():
            col += int(c)
    return torch.from_numpy(matrix_pos)

#we will use the probability to win, so we convert the evaluation in this form
def eval2score(evaluation):

    #if there is a mate combination we need to convert it
    if evaluation[0] =="#":
        mate_moves = int(evaluation[1:])
        cp = (21 - min(10, abs(mate_moves))) * 100
    else:
        cp = np.float(evaluation)
    score = 1/(1 + np.exp(-0.004 * cp))
    return score

#reverse the score if black active
def convert_evaluation(fen, evaluation):
    score = eval2score(evaluation)
    if fen.split()[1]=='b':
        return 1 - score
    return score


class ChessPotitionsDataset(Dataset):
    """Evaluated chess positions dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with positions.
        """
        self.positions = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.positions)//2

    def __getitem__(self, idx):
        fen = self.positions.iloc[2*idx]["FEN"]
        evaluation = self.positions.iloc[2*idx]["Evaluation"]
        matrix_pos = fen2matrix(fen)
        score = convert_evaluation(fen, evaluation)
        #convert to tensor for compatibility (we need float32, but standard is float64)
        return matrix_pos, torch.Tensor([score])


#Define the neuron net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(12, 64, 15, padding=7)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.conv2 = nn.Conv2d(64, 128, 15, padding=7)
        #self.bn2 = nn.BatchNorm2d(128)
        #self.conv3 = nn.Conv2d(128, 64, 15, padding=7)
        #self.bn3 = nn.BatchNorm2d(64)
        #self.conv4 = nn.Conv2d(64, 12, 15, padding=7)
        #self.bn4 = nn.BatchNorm2d(12)

        #convoluzione (numero channel input, numero channel output, dimensione matrice convoluzione, padding)
        self.fc1 = nn.Linear(12 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 500)
        self.fc3 = nn.Linear(500, 84)
        self.fc4 = nn.Linear(84, 1)

        for m in self.children():
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.bn1(x)
        #batch normalization
        #x = F.relu(x)
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = F.relu(x)
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = F.relu(x)
        #x = self.conv4(x)
        #x = self.bn4(x)
        #x = F.relu(x)

        #linearizzare il tensore per passare al full connected
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = F.sigmoid(x)
        return x


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    #dataset = pd.read_csv("dataset/chessData.csv", index_col=0)
    #baby_train = dataset[1:200]
    #baby_train.to_csv("dataset/baby_train.csv")
    dataset = pd.read_csv("dataset/baby_train.csv")

    #Use gpu for better performance
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tqdm.pandas()
    dataset["matrix_pos"] = dataset.FEN.progress_apply(fen2matrix)
    dataset['score'] = dataset.apply(lambda x: convert_evaluation(x.FEN, x.Evaluation), axis=1)
    dataset['score'] = dataset['score'].astype("float32")
    #board = chess.Board()
    ex_fen = 'r1bqkb1r/ppppnppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2' #str ci dice la posizione dei pezzi per riga # = spazio vuoto b= black turn
   # for i in range(len(dataset)):
    #    print(dataset.iloc[i]["score"])
    mat = fen2matrix(ex_fen)
    #board = chess.Board(ex_fen)
    #print(ex_fen.split()[1])
    #print(board)
    data_url = "dataset/chessData.csv"
    baby_url = "dataset/baby_train.csv"
    chess_dataset = ChessPotitionsDataset(csv_file=data_url)

    #num_workers è il numero di processori
    dataloader = DataLoader(chess_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=6)

    for i in range(10):
        print(chess_dataset[i])
    #mettere il net sul device
    net = Net().to(device)
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, score = data
            #score = score.view(-1, 1)
            #move tensors to gpu
            inputs = inputs.to(device)
            score = score.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print(outputs)

            loss = criterion(outputs, score)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("\r{}".format(loss.item()), end='\r')
            if i % 1000 == 0:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss*100))
                running_loss = 0.0
                torch.save(net, "/mnt/ramdisk/e{}-{}.pth".format(epoch, i))

    print('Finished Training')