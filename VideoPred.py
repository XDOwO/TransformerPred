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

video_len = 2
path = "/home/miles/桌面/inpaint/VideoPred/data/taiwan_monitor"
img_size=128 

EPOCH = 100000
BATCH= 64
device = "cuda"

    
dataset = VideoDataset(path,video_len,img_size)
dataloader = DataLoader(dataset,batch_size=BATCH,shuffle=True)

    
# model = VideoPredModel(3,64,512,3,3,10,6,6,10,0.1).to(device)
encoder = Encoder(output_channel=512,hidden_channel=128,num_layer=3).to("cuda")
decoder = Decoder(input_channel=512,hidden_channel=128,num_layer=3).to("cuda")
print(encoder)
print(decoder)
opt_encoder = optim.Adam(encoder.parameters(),lr=1e-4)
opt_decoder = optim.Adam(decoder.parameters(),lr=1e-4)
ssim = piqa.MS_SSIM().cuda()
mse = nn.MSELoss()
cl = nn.L1Loss()
# encoder.load_state_dict(th.load("encoder.pth"))
# decoder.load_state_dict(th.load("decoder.pth"))
criterion = piqa.MS_SSIM(window_size=7).cuda()


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
        if criterion is None:
            criterion = th.jit.trace(ssim, (data.view(-1,C,W,H), output.view(-1,C,W,H)))
        # print(data.min(),output.min())
        if isinstance(criterion,nn.MSELoss):
            loss = criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
        # else:
        #     loss = 1-criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
        # print(loss)
        loss = mse(data.view(-1,C,W,H),output.view(-1,C,W,H))+1-criterion(data.view(-1,C,W,H),output.view(-1,C,W,H))
        loss.backward()
        opt_encoder.step()
        opt_decoder.step()
        total_loss+=loss.item()
        # print(loss)
        # break
    print("Epoch:{},Loss:{}".format(epoch+1,total_loss))
    if (epoch+1)%100==0:
        encoder.eval()
        decoder.eval()
        data=dataset[0].view(1,*dataset[0].shape)
        # transforms.ToPILImage()(data.to("cpu")[0][0]).show()
        output = decoder(encoder(data.to(device)))
        # print(output.to("cpu")[0][0])
        # transforms.ToPILImage()(output.to("cpu")[0][0]).show()
        transforms.ToPILImage()(output.to("cpu")[0][0]).save("{}.jpg".format(epoch+1))
        print("Image Saved!")
            
    if epoch%100==0:
        th.save(encoder.state_dict(),"encoder.pth")
        th.save(decoder.state_dict(),"decoder.pth")

