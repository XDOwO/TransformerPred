import glob
import PIL.Image as Image
img=Image.open(glob.glob("./*.png")[0])
for i in range(1,9):
    img.save("{}.jpg".format(i))