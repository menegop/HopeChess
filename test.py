import torch
from main import Net, fen2matrix
model_path = "/mnt/ramdisk/e0-4000.pth"
net = torch.load(model_path)
fen_1 = "rnbqkbnr/ppp2ppp/8/4p3/2BpP3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4" #(a una mossa dal matto per il bianco)
fen_0 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" #(dopo la prima mossa)
fen_2 = "rnbqkbnr/pp1p2pp/8/1p2p3/4P1p1/3P1P2/KKKKKKKK/QQQQQQQQ w KQkq - 0 6"
#net.load_state_dict(torch.load(model_path))
net.eval()

in_ten = torch.zeros(100, 12, 8, 8).cuda()
in_ten[0] = fen2matrix(fen_2).unsqueeze_(0).cuda()
print(fen2matrix(fen_0))

out = net(in_ten)
print(out[0])