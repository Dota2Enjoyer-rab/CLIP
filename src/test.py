import pandas as pd
import os
import numpy as np
from PIL import Image
import pyarrow as pa

path = r"C:\Users\VN\Desktop\diploma\CLIP\data\cifar-10-batches-py\data_batch_3"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data = unpickle(path)

print(data.keys())
frame = data[b'data'][0]
print(data[b'labels'][-20])

# 'top_activations/img_7005.png', 'top_activations/img_6830.png', 'top_activations/img_7819.png', 'top_activations/img_241.png', 'top_activations/img_5051.png', 'top_activations/img_8123.png', 'top_activations/img_9658.png', 'top_activations/img_2346.png' - грузовки с высокой вероятностью 
# 0,"['top_activations/img_5399.png', 'top_activations/img_8634.png', 'top_activations/img_5216.png', 'top_activations/img_1661.png', 'top_activations/img_7123.png']","[7.0557403564453125, 6.97554874420166, 6.787762641906738, 6.732084274291992, 6.581116676330566]",0.5436981320381165,7.0557403564453125
#"['top_activations_live/img_6133.png', 'top_activations_live/img_8960.png', 'top_activations_live/img_6821.png', 'top_activations_live/img_9017.png', 'top_activations_live/img_7116.png', 'top_activations_live/img_3782.png', 'top_activations_live/img_9482.png', 'top_activations_live/img_4546.png']
# 2047,7.0386152267456055,0.1284237951040268,"['top_activations_live/img_7632.png', 'top_activations_live/img_3684.png', 'top_activations_live/img_7994.png', 'top_activations_live/img_2821.png', 'top_activations_live/img_4840.png', 'top_activations_live/img_4374.png', 'top_activations_live/img_5302.png', 'top_activations_live/img_4802.png']","[7.0386152267456055, 6.180198669433594, 6.132760524749756, 5.928404808044434, 5.845181941986084, 5.6085052490234375, 5.505102157592773, 5.292472839355469]"
# 4043,4.795070171356201,0.2916426658630371,"['top_activations_live/img_1700.png', 'top_activations_live/img_1362.png', 'top_activations_live/img_7540.png', 'top_activations_live/img_6259.png', 'top_activations_live/img_1055.png', 'top_activations_live/img_3204.png', 'top_activations_live/img_903.png', 'top_activations_live/img_3230.png']","[4.795070171356201, 4.682876110076904, 4.427708148956299, 4.1997880935668945, 4.192198276519775, 4.174424171447754, 4.141254425048828, 4.111608505249023]"
# 40,8.021261215209961,0.4132240116596222,"['top_activations_live/img_7005.png', 'top_activations_live/img_6830.png', 'top_activations_live/img_7819.png', 'top_activations_live/img_241.png', 'top_activations_live/img_5051.png', 'top_activations_live/img_8123.png', 'top_activations_live/img_9658.png', 'top_activations_live/img_2346.png']","[8.021261215209961, 7.3138203620910645, 7.089373588562012, 6.943542957305908, 6.91518497467041, 6.83616304397583, 6.793586730957031, 6.684114933013916]"
img_array = frame.reshape(3, 32, 32).transpose(1, 2, 0)
img = Image.fromarray(img_array.astype('uint8'), mode="RGB")
img.show()
