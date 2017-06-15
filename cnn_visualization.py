# https://raghakot.github.io/keras-vis/visualizations/attention/#overview
# https://github.com/jacobgil/keras-cam/blob/master/model.py#L59

from false_reduction_cnn import sealion_vgg16, sealion_cnn

import numpy as np
from matplotlib import pyplot as plt
import h5py

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency, visualize_cam


'''
# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
print('Model loaded.')

image_paths = [
    "http://www.tigerfdn.com/wp-content/uploads/2016/05/How-Much-Does-A-Tiger-Weigh.jpg",
    "http://www.slate.com/content/dam/slate/articles/health_and_science/wild_things/2013/10/131025_WILD_AdeliePenguin.jpg.CROP.promo-mediumlarge.jpg",
    "https://www.kshs.org/cool2/graphics/dumbbell1lg.jpg",
    "http://tampaspeedboatadventures.com/wp-content/uploads/2010/10/DSC07011.jpg",
    "http://ichef-1.bbci.co.uk/news/660/cpsprodpb/1C24/production/_85540270_85540265.jpg"
]
'''

# Set variables
K.set_image_data_format('channels_first') # this for visualize_saliency, visualize_cam to get the right image shape
visual_type = 'sal' # or 'cam'
weight_dir = "/home/hao/Desktop/sealion_training_data/cnn_small_sample/sealion_magic1.h5"
cnn_sample_txt_dir = "/home/hao/Desktop/sealion_training_data/cnn_small_sample/test.txt" 
target_img_size = 64
num_class = 5

# Init model
model = sealion_vgg16(num_class)
model.summary()

# Load weight
f = h5py.File(weight_dir)
print(f.attrs.keys()) # [u'keras_version', u'backend', u'model_config', u'training_config']
print("*********************************************")
model.load_weights(weight_dir)
'''
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        break # we don't look at the last (fully-connected) layers in the savefile
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
    model.layers[k].trainable = False
f.close()
'''
# The name of the layer we want to visualize; hard coding here, better to always name the last layer 'predictions'
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to different species of sealion, read it from cnn_sample.txt
image_paths = [line.strip() for line in open(cnn_sample_txt_dir)]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(path, target_size=(target_img_size, target_img_size))
    x = np.expand_dims(img_to_array(seed_img), axis=0) # already set_image_data_format('channels_first'), hence no need to transpose data
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))
    print("INFO: Input image shape is " + str(seed_img.shape))
    print("INFO: Pred class for this image is %s" % pred_class)
    print("INFO: Visualizing image of layer_idx = %s" % layer_idx)

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    if visual_type == 'sal':
        heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
    elif visual_type == 'cam':
        heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img, None, 0.4)
    else:
        raise ValueError("Error visual_type, Exit.")
    heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Visualization Map')
plt.show()
