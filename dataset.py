from torch.utils.data import Dataset
from torchvision import transforms
import torch as th
from PIL import Image
import glob
class VideoDataset(Dataset):
    def __init__(self,path,video_len,size):
        super().__init__()
        self.img_paths=glob.glob(path+"/*.jpg")
        self.path = path
        self.size = size
        self.video_len = video_len
        self.transform = transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()])
        self.sample={}
    def __getitem__(self, index):
        try:
            return self.sample[index]
        except:
            pass
        video = th.Tensor()
        for i in range(self.video_len):
            img = Image.open(self.img_paths[index*self.video_len+i]).convert("RGB")
            imgt = self.transform(img)
            imgt = th.unsqueeze(imgt,0)
            video = th.cat((video,imgt))
        self.sample[index] = video
        return video

    def __len__(self):
        return len(self.img_paths)//self.video_len