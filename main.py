# This is a sample Python script.

import chess
import numpy as np
# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from skimage import io, transform
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#convert fen string to binary array (JUST A TEST)
def get_extended_positions(fen):
    ext_pieces_positions = np.zeros([2,6, 8, 8])
    pieces = list("RNBQKPrnbqkp")
    col = 0
    row = 0
    for c in fen:
        for i, p in enumerate(pieces):
            if c == p:
                ext_pieces_positions[i//6, i%6, row, col] = 1
                col += 1
                break
        if c == '/':
            row += 1
            col = 0
        elif c.isdigit():
            col += int(c)
        elif c == ' ':
            return ext_pieces_positions

#convert fen string to integer array (JUST A TEST)
def get_pieces_positions(fen):
    pieces_positions = np.zeros([32, 2])
    pieces = list("RRNNBBQKPPPPPPPPrrnnbbqkpppppppp")
    col = 1
    row = 1
    for c in fen:
        for p in range(32):
            if c == pieces[p]:
                pieces[p] = 'Z'
                pieces_positions[p] = [col, row]
                col += 1
                break
        if c == '/':
            row += 1
            col = 1
        elif c.isdigit():
            col += int(c)
        elif c == ' ':
            return pieces_positions

#convert fen string to binary array (THIS IS THE GOOD FUNCTION)
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
        cp = float(evaluation)
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
        return len(self.positions)

    def __getitem__(self, idx):
        fen = self.positions.iloc[idx]["FEN"]
        evaluation = self.positions.iloc[idx]["Evaluation"]
        matrix_pos = fen2matrix(fen)
        score = convert_evaluation(fen, evaluation)
        return matrix_pos, score

#Define the neuron net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    #dataset = pd.read_csv("dataset/chessData.csv", index_col=0)
    #baby_train = dataset[1:200]
    #baby_train.to_csv("dataset/baby_train.csv")
    dataset = pd.read_csv("dataset/baby_train.csv")



    tqdm.pandas()
    dataset["matrix_pos"] = dataset.FEN.progress_apply(fen2matrix)
    dataset['score'] = dataset.apply(lambda x: convert_evaluation(x.FEN, x.Evaluation), axis=1)
    dataset['score'] = dataset['score'].astype("float32")
    print(dataset.dtypes)
    #board = chess.Board()
    ex_fen = 'r1bqkb1r/ppppnppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2'
   # for i in range(len(dataset)):
    #    print(dataset.iloc[i]["score"])
    mat = fen2matrix(ex_fen)
    print(mat.dtype)
    #board = chess.Board(ex_fen)
    #print(ex_fen.split()[1])
    #print(board)
    data_url = "dataset/chessData.csv"
    baby_url = "dataset/baby_train.csv"
    chess_dataset = ChessPotitionsDataset(csv_file=baby_url)
    dataloader = DataLoader(chess_dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    for i in range(2):
        print(chess_dataset[i])
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')