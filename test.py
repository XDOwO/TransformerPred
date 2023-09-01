import torch as th
import torch.nn as nn
import torch.functional as f
from random import randint
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from model import VideoPredModel, Encoder,Decoder,TempAutoEncoder
import glob
import torch.optim as optim
import piqa
from dataset import VideoDataset
import matplotlib.pyplot as plt
video_len = 2
path = "/home/miles/桌面/inpaint/guided-diffusion/video_img"
img_size=128
EPOCH = 1000000
BATCH= 4
device = "cuda"

    
dataset = VideoDataset(path,video_len,img_size)
dataloader = DataLoader(dataset,batch_size=BATCH,shuffle=True)

    
# model = VideoPredModel(3,64,512,3,3,10,6,6,10,0.1).to(device)
encoder = Encoder(output_channel=512,hidden_channel=128,num_layer=3).to("cuda")
decoder = Decoder(input_channel=512,hidden_channel=128,num_layer=3).to("cuda")
model = TempAutoEncoder().to("cuda")
# opt = optim.Adam(model.parameters(),lr=1e-4)
ssim = piqa.MS_SSIM(7).cuda()
criterion = None # nn.MSELoss()



target = 1000
# from tqdm import tqdm
# for epoch in range(EPOCH):
#     total_loss = 0
#     model.train()
#     for data in tqdm(dataloader,desc="Epoch:{}".format(epoch+1)):
#         N,T,C,W,H = data.shape
#         data = data.to(device)
#         data.requires_grad_(True)
#         opt.zero_grad()
#         output = model(data)
#         if criterion is None:
#             criterion = th.jit.trace(ssim, (data.view(-1,C,W,H), output.view(-1,C,W,H)))
#         # print(data.min(),output.min())
#         if isinstance(criterion,nn.MSELoss):
#             loss = criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
#         else:
#             loss = 1-criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
#         # print(loss)
#         loss.backward()
#         opt.step()
#         total_loss+=loss.item()
#         # print(loss)
#         # break
#     print("Epoch:{},Loss:{}".format(epoch+1,total_loss))
#     if (epoch+1)%5==0:
#         model.eval()
#         data=dataset[0].view(1,*dataset[0].shape)
#         # transforms.ToPILImage()(data.to("cpu")[0][0]).show()
#         output = model(data.to(device))
#         # print(output.to("cpu")[0][0])
#         transforms.ToPILImage()(output.to("cpu")[0][0]).save("{}.jpg".format(epoch+1))
#         print("Image Saved!")
            
#     if epoch%20==0:
#         th.save(model.state_dict(),"model.pth")
# model.eval()
encoder.load_state_dict(th.load("encoder.pth"))
decoder.load_state_dict(th.load("decoder.pth"))
# model.load_state_dict(th.load("model.pth"))
i=0
for data in dataloader:
    N,T,C,W,H = data.shape
    # output = model(data.to(device))
    output = decoder(encoder(data.to(device)))
    # print(output.to("cpu")[0][0])
    for x in range(video_len):
        # print(x*10+i)
        img1 = transforms.ToPILImage()(output.to("cpu")[0][x])
        img2 = transforms.ToPILImage()(data.to("cpu")[0][x])
        plt.figure(figsize=(10,10))
        plt.subplot(221)
        plt.imshow(img1)
        plt.subplot(222)
        plt.imshow(img2)
        plt.savefig('{}.png'.format(x+i*video_len))
    print(ssim(data.to(device).view(-1,C,W,H),output.view(-1,C,W,H)).item())
    i+=1
    if i==5:
        break
