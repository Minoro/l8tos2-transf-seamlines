import osgeo.gdal as gdal
import os, sys
from glob import glob
import numpy as np
import math
import pandas as pd
from tqdm import tqdm


IMAGES_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/stack'
OUTPUT_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/imgs_256'

CROP_STEP = 256 * 2


def get_extent(dataset):

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()

    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]
    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]

    return {"minX": str(minx), "maxX": str(maxx),
            "minY": str(miny), "maxY": str(maxy),
            "cols": str(cols), "rows": str(rows)}


def get_split(fileIMG,out_path):
    
    dataset = gdal.Open(fileIMG)
    # mask = dataset.GetRasterBand(1).ReadAsArray()


    passo = CROP_STEP
    xsize = 1*passo
    ysize = 1*passo

    extent = get_extent(dataset)
    cols = int(extent["cols"])
    rows = int(extent["rows"])

    nx = (math.ceil(cols/passo))
    ny = (math.ceil(rows/passo))
    
    #print(nx*ny)

    cont = 0

    for i in range(0,nx):
        for j in range(0,ny):
            xoff = passo*i
            yoff = passo*j
            
            if cols-xoff < passo or rows-yoff < passo:
                continue

            cont += 1
            dst_dataset = os.path.join(out_path, os.path.basename(fileIMG)[:-4]+'_p'+str(cont).zfill(5)+'.tif')


            if not os.path.exists(dst_dataset):
            
                # if xoff+xsize > cols: 
                #     n2 = range(xoff,cols)
                # else:
                #     n2 = range(xoff,xoff+xsize)                    
                    
                # if yoff+ysize > rows: 
                #     n1 = range(yoff,rows) 
                # else:
                #     n1 = range(yoff,yoff+ysize)
                    
                # if np.amax(mask[np.ix_(n1,n2)]):
                #     contp += 1
                #     gdal.Translate(dst_dataset, dataset, srcWin = [xoff, yoff, xsize, ysize])                
            
                gdal.Translate(dst_dataset, dataset, srcWin = [xoff, yoff, xsize, ysize])                
                

    return cont

if __name__ == '__main__':

    images = glob(os.path.join(IMAGES_DIR, '*.tif'))

    print(f'Num Images: {len(images)}')


    print('Cropando imagens rgb anotadas manualmente')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for image in tqdm(images, total=len(images)):
        get_split(image, OUTPUT_DIR)

    print('Done!')
    
        