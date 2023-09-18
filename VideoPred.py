import torch as th
import torch.nn as nn
import torch.functional as f
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from model import VideoPredModel,Encoder,Decoder
import glob
import torch.optim as optim
import piqa
from dataset import VideoDataset
import matplotlib.pyplot as plt
video_len = 2
path = "C:/Users/許廷豪/data/taiwan_monitor"
# path = "C:/Users/許廷豪/data/onlytwo"
img_size= 128

EPOCH = 100000
BATCH= 32
device = "cuda"

    
dataset = VideoDataset(path,video_len,img_size)
dataloader = DataLoader(dataset,batch_size=BATCH,shuffle=True)

    
# model = VideoPredModel(3,64,512,3,3,10,6,6,10,0.1).to(device)
encoder = Encoder(output_channel=128,hidden_channel=128,num_layer=3).to("cuda")
decoder = Decoder(input_channel=128,hidden_channel=128,num_layer=3).to("cuda")
#8192
# print(encoder)
# print(decoder)
opt_encoder = optim.Adam(encoder.parameters(),lr=1e-4)
opt_decoder = optim.Adam(decoder.parameters(),lr=1e-4)
scheduler_encoder = th.optim.lr_scheduler.ReduceLROnPlateau(opt_encoder, factor=0.1, patience=10)
scheduler_decoder = th.optim.lr_scheduler.ReduceLROnPlateau(opt_decoder, factor=0.1, patience=10)
ssim = piqa.SSIM().cuda()
mse = nn.MSELoss()
mae = nn.L1Loss(reduction="mean")
cl = nn.L1Loss()
encoder.load_state_dict(th.load("encoder.pth"))
decoder.load_state_dict(th.load("decoder.pth"))
criterion = None #piqa.MS_SSIM(7).cuda()


from tqdm import tqdm
for epoch in range(EPOCH):
    total_loss = 0
    encoder.train()
    decoder.train()
    for data in tqdm(dataloader,desc="Epoch:{}".format(epoch+1)):
        N,T,C,W,H = data.shape
        data = data.to(device)
        data.requires_grad_(True)
        data.retain_grad()
        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        output = decoder(encoder(data))
        # if criterion is None:
        #     criterion = th.jit.trace(ssim, (data.view(-1,C,W,H), output.view(-1,C,W,H)))
        # print(data.min(),output.min())
        # if isinstance(criterion,nn.MSELoss):
        #     loss = criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
        # else:
        #     loss = 1-criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
        # print(loss)
        loss = mse(data.view(-1,C,W,H),output.view(-1,C,W,H))#1-criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
        loss.backward()
        opt_encoder.step()
        opt_decoder.step()
        total_loss+=loss.item()
        
        # print(loss)
        # break
    print("Epoch:{},Loss:{}".format(epoch+1,total_loss))
    encoder.eval()
    decoder.eval()
        
    if (epoch+1)%100==0:
        img1 = transforms.ToPILImage()(output.to("cpu")[0][0])
        img2 = transforms.ToPILImage()(data.to("cpu")[0][0])
        plt.figure(figsize=(10,10))
        plt.subplot(221)
        plt.imshow(img1)
        plt.subplot(222)
        plt.imshow(img2)
        plt.savefig('{}.png'.format(0))
        # transforms.ToPILImage()(output.to("cpu")[0][0]).save("{}.jpg".format(epoch+1))
        print("Image Saved!")
        plt.close()
            
    if (epoch+1)%100==0:
        th.save(encoder.state_dict(),"encoder.pth".format(epoch+1))
        th.save(decoder.state_dict(),"decoder.pth".format(epoch+1))

