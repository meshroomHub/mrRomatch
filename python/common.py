from pyalicevision import image as avimage
from pyalicevision import sfmData as avsfmData
from pyalicevision import sfmDataIO as avsfmDataIO
from pyalicevision import matchingImageCollection as avmic   

import numpy as np
from PIL import Image

class ImageInfo:
    def __init__(self):
        self.path = ""
        self.uid = 0
        self.width = 0
        self.height = 0

def open_image(path, isBW = False, isFloat = True):
    """ Load an image using pyalicevision

    Parameters:
    path : the filesystem path to the image
    isBW : should we load the image as a 1 channel, grayscale image
    isFloat : does it use float or unsigned char per channel ?
    """

    image = []
    if isBW:
        if isFloat:
            image = avimage.Image_float()
        else:
            image = avimage.Image_uchar()
    else:
        if isFloat:
            image = avimage.Image_RGBfColor()
        else:
            image = avimage.Image_RGBColor()

    optRead = avimage.ImageReadOptions(avimage.EImageColorSpace_NO_CONVERSION)
    avimage.readImage(path, image, optRead)

    return image

def open_image_to_pil(path):
    """ Load an image using pyalicevision

    Parameters:
    path : the filesystem path to the image
    """

    image = avimage.Image_RGBColor()
    optRead = avimage.ImageReadOptions(avimage.EImageColorSpace_NO_CONVERSION)
    avimage.readImage(path, image, optRead)

    return Image.fromarray(image.getNumpyArray())

def open_image_as_numpy(path, isBW = False, isFloat = True):
    """ Load an image using pyalicevision

    Parameters:
    path : the filesystem path to the image
    isBW : should we load the image as a 1 channel, grayscale image
    isFloat : does it use float or unsigned char per channel ?
    """
    if path is None:
        return None
        
    image = open_image(path, isBW, isFloat)
    return image.getNumpyArray()

def save_image(path, array, isBW = False):
    """ Save an image using pyalicevision

    Parameters:
    path : the filesystem path to the image
    array : the numpy array content of the image
    isBW : should we load the image as a 1 channel, grayscale image
    """
    
    image = []
    if array.dtype == np.uint8:
        if isBW:
            image = avimage.Image_uchar()
        else: 
            image = avimage.Image_RGBColor()
    else:
        if isBW:
            image = avimage.Image_float()
        else: 
            image = avimage.Image_RGBfColor()
    
    image.fromNumpyArray(array)

    optWrite = avimage.ImageWriteOptions()
    optWrite.toColorSpace(avimage.EImageColorSpace_NO_CONVERSION)
    avimage.writeImage(path, image, optWrite)

def get_imageinfos_from_sfmdata(inputSfMData):
    """
    Parse a sfmData and fill a list of ImageInfos struct
    
    Parameters:
    inputSfMData : the filesystem path to the sfmData
    
    Returns:
    a dict (indexed by view uid) of ImageInfo descriptions
    """
    
    data = avsfmData.SfMData()
    ret = avsfmDataIO.load(data, inputSfMData, avsfmDataIO.VIEWS)
    if not ret:
        raise RuntimeError("Error with sfm data")
        
    views = data.getViews()
    
    nb_images = len(views)
    
    #sort by frameid
    views = dict(sorted(views.items(), key=lambda v: v[1].getFrameId()))
    
    # utils variables
    infos = dict()
    for (key, view) in views.items():
        item = ImageInfo()
        item.uid = key
        item.path = view.getImage().getImagePath()
        item.width = int(view.getImage().getWidth())
        item.height = int(view.getImage().getHeight())
        infos[key] = item

    return infos

def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")