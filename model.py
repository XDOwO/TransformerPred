import math
import torch as th
from torchvision import transforms
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(ResBlock,self).__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.cnn = nn.Conv2d(input_channel,output_channel,3)
        self.bn = nn.BatchNorm2d(output_channel)
    def forward(self,x):
        return self.bn(x + self.cnn(self.pad(x)))

class EncoderBlock(nn.Module):
    def __init__(self,input_channel=3,output_channel=64,kernel_size=3,num_layer=1,stride=2,padding=1,img_size=128):
        super(EncoderBlock,self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.num_layer = num_layer
        self.c_dm = img_size*img_size//4
        self.resblock=nn.ModuleList([nn.Conv2d(input_channel,output_channel,kernel_size,stride,padding),nn.BatchNorm2d(output_channel),nn.ReLU()])#self-attention is here if you need nn.MultiheadAttention(output_channel,4,batch_first=True)
        self.conv2d = nn.Conv2d(input_channel,output_channel,kernel_size,stride=2,padding=1)
        self.qconv=nn.Conv2d(output_channel,output_channel//1,1)
        self.kconv=nn.Conv2d(output_channel,output_channel//1,1)
        self.vconv=nn.Conv2d(output_channel,output_channel,1)
        self.qconv2=nn.Conv1d(self.c_dm ,self.c_dm //1,1)
        self.kconv2=nn.Conv1d(self.c_dm ,self.c_dm //1,1)
        self.vconv2=nn.Conv1d(self.c_dm ,self.c_dm ,1)
        self.bn=nn.BatchNorm2d(output_channel)

    def forward(self,x):
        ox = x.detach().clone()
        d = 0
        whx = 0
        for f in self.resblock:
            if isinstance(f,nn.MultiheadAttention) and d==0:
                
                d+=1
                q,k,v = self.qconv(x),self.kconv(x),self.vconv(x)
                N,C_QK,W,H = q.shape
                _,C_V ,_,_ = v.shape
                q=q.view(N,C_QK,W*H).permute(0,2,1)
                k=k.view(N,C_QK,W*H).permute(0,2,1)
                v=v.view(N,C_V,W*H).permute(0,2,1)
                whx,_=f(q,k,v)
                whx=whx.permute(0,2,1).view(N,C_V,W,H)
            elif isinstance(f,nn.MultiheadAttention):
                px = x.permute(0,2,3,1).view(N,W*H,-1)
                q,k,v = self.qconv2(px),self.kconv2(px),self.vconv2(px)
                N,WH,C_QK = q.shape
                _, _,C_V  = v.shape
                q=q.permute(0,2,1)
                k=k.permute(0,2,1)
                v=v.permute(0,2,1)
                cx,_=f(q,k,v)
                cx=cx.view(N,C_V,W,H)
            else:
                x = f(x)
        return self.bn(x) #+cx +self.conv2d(ox)

class Encoder(nn.Module):
    def __init__(self,img_size=128,input_channel=3,hidden_channel=16,output_channel=64,kernel_size=3,num_layer=1,stride=2,padding=1,resblock_num=8):
        super(Encoder,self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.num_layer = num_layer
        if num_layer==0:
            self.module = [EncoderBlock(input_channel,output_channel,kernel_size,stride=stride,padding=padding,img_size=img_size)]
        else:
            self.module = [EncoderBlock(input_channel,hidden_channel,kernel_size,stride=stride,padding=padding,img_size=img_size)]+\
                                     [EncoderBlock(hidden_channel,hidden_channel,kernel_size,stride=stride,padding=padding,img_size=img_size//2**(i+1)) for i in range((num_layer-1))]+\
                                     [EncoderBlock(hidden_channel,output_channel,kernel_size,stride=stride,padding=padding,img_size=img_size//2**num_layer)]
        self.module += [ResBlock(output_channel,output_channel)]*resblock_num
        self.module = nn.Sequential(*self.module)
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.normal_(layer.weight)
    def forward(self,x):
        x=x.flatten(0,1)
        return self.module(x)

class DecoderBlock(nn.Module):
    def doing_nothing(self,x):
        return 0
    def __init__(self,input_channel,output_channel,kernel_size,stride=2,padding=1,have_relu=True,upsample="scale"):
        super(DecoderBlock,self).__init__()
        if upsample=="t_conv":
            self.t_conv=nn.Sequential(nn.ConvTranspose2d(input_channel,output_channel,kernel_size,stride,padding,output_padding=1))
            self.scale=self.doing_nothing
        elif upsample=="scale":
            self.upsample = nn.Upsample(scale_factor=2,mode = 'bilinear')
            self.pad = nn.ReplicationPad2d(1)
            self.conv = nn.Conv2d(input_channel,output_channel, kernel_size=3, stride=1, padding=0)
            self.scale=nn.Sequential(self.upsample,self.pad,self.conv)
            self.t_conv=self.doing_nothing
        else:
            self.t_conv=nn.Sequential(nn.ConvTranspose2d(input_channel,output_channel,kernel_size,stride,padding,output_padding=1))
            self.scale=nn.Sequential(nn.Upsample(scale_factor=2),nn.ReplicationPad2d(1),nn.Conv2d(input_channel,output_channel,kernel_size))
        self.bn=nn.BatchNorm2d(output_channel)
        self.relu=nn.ReLU()
        # self.nn_size = list(zip(nn_size[:-1],nn_size[1:]))
        # self.nn = nn.ModuleList([nn.Linear(x,y) for x,y in self.nn_size])
        self.have_relu=have_relu
    def forward(self,x):

        if self.have_relu:
            x=self.relu(self.bn(self.scale(x)))
        else:
            x=self.bn(self.scale(x))
        # print(x.shape)
        # B,C,W,H = x.shape
        # x=x.view(B,C*W*H)
        # for f in self.nn:
        #     print(x.shape,self.nn_size)
        #     x = f(x)
        #     x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self,image_size=128,input_channel=64,hidden_channel=16,output_channel=3,kernel_size=3,num_layer=1,stride=2,padding=1,video_len=2):
        super(Decoder,self).__init__()
        self.image_size = image_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.num_layer = num_layer
        self.nnlist=[(image_size//2**(num_layer+1))**2*input_channel,1,(image_size**2)*3]
        self.nnlist=list(zip(self.nnlist[:-1],self.nnlist[1:]))
        if num_layer == 0:
            self.transposed_cnn = nn.ModuleList([DecoderBlock(input_channel,output_channel,kernel_size,stride,padding,have_relu=False)])
        else:
            self.transposed_cnn = nn.ModuleList([DecoderBlock(input_channel,hidden_channel,kernel_size,stride,padding)]+
                                                [DecoderBlock(hidden_channel,hidden_channel,kernel_size,stride,padding) for i in range(num_layer-1)]+
                                                [DecoderBlock(hidden_channel,output_channel,kernel_size,stride,padding,have_relu=False)]
                                                )
        # self.nns =nn.ModuleList([i for sublist in [[nn.Linear(i,o),nn.BatchNorm1d(o)] for i,o in self.nnlist] for i in sublist ])
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.last_cnn=nn.Sequential(nn.ReplicationPad2d(3),nn.Conv2d(output_channel,output_channel,7,padding=0))
        # self.linear=nn.Sequential(nn.Linear(3*128*128,4096),nn.Linear(4096,3*128*128))
        self.video_len = video_len
        for layer in self.modules():
            if isinstance(layer,nn.ConvTranspose2d):
                nn.init.normal_(layer.weight)
    def forward(self,x):
        for f in self.transposed_cnn:
            x = f(x)
        # x = self.relu(x)
        # x = self.last_cnn(x)
        # x = x.view(B,C*W*H)
        # x = self.linear(x)
        # for f in self.nns[:-1]:
        #     x = f(x)
        #     x = self.relu(x)
        # x = self.nns[-1](x)
        x = self.sigmoid(x)
        return x.view(-1,self.video_len,3,self.image_size,self.image_size)





class PositionalEncoding(nn.Module):

    def __init__(self,dim):
        super().__init__()
        self.dim = dim
    def forward(self,time):
        device = time.device
        half_dim = self.dim//2
        embbedings = math.log(10000) / (half_dim-1)
        embbedings = th.exp(th.arange(half_dim,device = device)*-embbedings)
        embbedings = time[:, None] * embbedings[None , :]
        embbedings = th.cat((embbedings.sin(),embbedings.cos()),dim=-1)
        return embbedings

class VideoPredModel(nn.Module):
    def __init__(self,in_channel,hidden_channel,d_model,kernel_size,ED_layer_num,nhead,transformer_encoder_layer_num,transformer_decoder_layer_num,dim_ff,dropout):
        super().__init__()
        self.encoder = Encoder(in_channel,hidden_channel,d_model,kernel_size,ED_layer_num)
        self.decoder = Decoder(d_model,hidden_channel,in_channel,kernel_size,ED_layer_num)
        self.embedding = nn.Embedding() #TODO
        self.transformer = nn.Transformer(d_model,nhead,transformer_encoder_layer_num,transformer_decoder_layer_num,dim_ff,dropout,batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        
        self.PE = 0 #TODO
    
    def forward(self,x):
        N,T,C,H,W=x.size()
        x=x.flatten(0,1)
        x=self.encoder(x)
        x = self.ln(x)
        x=self.transformer(x[:,:T//2,:,:,:],x[:,T//2:,:,:,:])
        x=self.ln(x)
        x=self.decoder(x)
        x=self.ln(x)
        x=x.view(N,T//2,C,H,W)
        return x

class TempAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n=512
        self.encoder = Encoder(output_channel=self.n,hidden_channel=128,num_layer=3)
        self.decoder = Decoder(input_channel=self.n,hidden_channel=128,num_layer=3)
        self.bn=nn.BatchNorm2d(self.n)
        # self.nn1 = nn.Linear(self.n*8*8,100)
        # self.nn2 = nn.Linear(100,100)
        # self.nn3 = nn.Linear(100,self.n*8*8)
    def forward(self,x):
        N,T,C,H,W=x.size()
        x = x.flatten(0,1)
        x = self.encoder(x)
        # x = self.bn(x)
        # transforms.ToPILImage()(x[0]).show()
        # print(x.shape)
        # x = x.view(N*T,self.n*8*8)
        # x = self.nn1(x)
        # x = self.nn2(x)
        # x = self.nn3(x)
        # x = x.view(N*T,self.n,8,8)
        x = self.decoder(x)
        # print(x.shape)
        x = x.view(N,T,C,H,W)
        return x