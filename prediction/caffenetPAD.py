import tensorflow as tf
import numpy as np
import os.path
import PIL
from PIL import Image, ImageEnhance, ImageStat
import time
import math

tf.logging.set_verbosity(tf.logging.ERROR)
path_NNet = '../msh_tanzania_3k_12.nnet'
path_model = '../'
pad_shape = (227, 227)

def loadNN(nnet_file):
  with open(nnet_file) as fin:
      content = fin.readlines()
  fin.close
  content = [x.strip() for x in content]
  drugs = ""
  exclude = ""
  model_checkpoint = ""
  typ = ""

  for line in content:
      if 'DRUGS' in line:
          drugs = line[6:].split(',')
      elif 'LANES' in line:
          exclude = line[6:]
      elif 'WEIGHTS' in line:
          model_checkpoint = line[8:]
      elif 'TYPE' in line:
          typ = line[5:]

  #test for Tensorflow
  if typ != "tensorflow":
      print('Not tensorflow network!')
      return
  else:
      return drugs, exclude, model_checkpoint, typ

def identify(image, drugs, exclude, model_checkpoint, typ):
    #reshape the image to an (1, 154587) vector
    image = np.mat(np.asarray(image).flatten())
    #create session
    with tf.Session() as sess:
        #load in the saved weights
        model_loc = path_model + model_checkpoint
        saver = tf.train.import_meta_graph(model_loc +'.meta')
        saver.restore(sess, model_loc)

        #get graph to extract tensors
        graph = tf.get_default_graph()
        pred = graph.get_tensor_by_name("pred:0")
        output = graph.get_tensor_by_name("output:0")
        X = graph.get_tensor_by_name("X:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        #find the prediction
        result = sess.run(pred, feed_dict={X: image, keep_prob: 1.})
        #we can look at the softmax output as well
        prob_array = sess.run(output, feed_dict={X: image, keep_prob: 1.})
        #print("result", result,"prob",prob_array)       
        return result[0], prob_array.reshape(-1)
    

    
# function to return average brightness of an image
# Source: http://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
def brightness(im):
    stat = ImageStat.Stat(im)
    r,g,b = stat.mean
    #return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))   #this is a way of averaging the r g b values to derive "human-visible" brightness
    return math.sqrt(0.577*(r**2) + 0.577*(g**2) + 0.577*(b**2))

# Defining the resize function to resize images in folder
def resize_pad(original, size, fix_bright = True):
    
    if fix_bright:
        # Fix brightness
        bright = brightness(original)

        # Massage image
        imgbright = ImageEnhance.Brightness(original)
        original = imgbright.enhance(165.6/bright)
    
    # For square images  
    im = original.resize((size), Image.ANTIALIAS)
 
    return im 

def cut_pad(img, exclude = ""):    
    # crop out active area
    img = img.crop((71, 359, 71+636, 359+490))
    
    # lanes split
    lane = []

    # exclude these lanes, now loaded from nnet file
    # exclude = "AJ"

    #loop over lanes
    for i in range(0,12):
        if chr(65+i) not in exclude:
            lane.append(img.crop((53*i, 0, 53*(i+1), 490)))

    # reconstruct
    imgout = Image.new("RGB", (53 * len(lane), 490))

    # loop over lanes
    for i in range(0,len(lane)):
        imgout.paste(lane[i], (53*i, 0, 53*(i+1), 490))

    #resize
    imgout = imgout.resize(pad_shape, Image.ANTIALIAS)

    return imgout

def processImage(image_loc, exclude):
    
    # open image
    img = PIL.Image.open(image_loc)
   
    # check image rectified
    width, height = img.size
    
    if (width != pad_shape[0]) & (height != pad_shape[1]):
        if width == 730 or height == 1220:
            img = cut_pad(img)

        elif (width == 636) & (height == 490):
            img = resize_pad(img, pad_shape)
            
        elif (width == 256) & (height == width):
            img = resize_pad(img, pad_shape, fix_bright = False)
            
        else:
            print("Image not rectified or out of expected PAD shape.")

    return img

def predict(image_loc):

    # Load caffenet
    drugs, exclude, model_checkpoint, typ = loadNN(path_NNet)
    
    # Process image
    img = processImage(image_loc, exclude)
   
    # Predict drug
    pd, pp = identify(img, drugs, exclude, model_checkpoint, typ)
    return drugs[pd], pd, pp[pd], pp

if __name__ == '__main__':
    predict("./data/test.jpg")
