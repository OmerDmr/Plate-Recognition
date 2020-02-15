from procesAndSelectClassifier import extractFeature
import pickle
import cv2
import os
#from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from timeit import default_timer as timer

color_space = 'YCrCb'
orient = 9
pix_per_cell = 8 
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

buffer_weights=[0.1,0.2,0.3,0.4]
heatmap_buffer = []
N_buffer = 3
y_start_stop = [400, 656]
ystart_0 = y_start_stop[0]
ystop_0 = ystart_0 + 64*2
ystart_1 = ystart_0
ystop_1 = y_start_stop[1]
ystart_2 = ystart_0
ystop_2 = y_start_stop[1]
ystarts = [ystart_1, ystart_2]
ystops = [ystop_1-100, ystop_2]
windowScales = [1.5, 2] # (64x64), (96x96), (128x128)

modelPath= 'classifier.p'

if os.path.isfile(modelPath):
	print("Loading existing Model")
	with open('classifier.p', 'rb') as f:
		model = pickle.load(f)
	X_scaler = model['scaler']
	cls = model['classifier']
else:

	trainAndTest()
	


def slidingWindows(img, x_start_stop=[None, None], y_start_stop=[None, None],
			 window=(64, 64), overlap=(0.5, 0.5)):
	x_start_stop[0] = 0
        x_start_stop[1] = img.shape[1]
        y_start_stop[0] = 0
        y_start_stop[1] = img.shape[0]
  	xspan = x_start_stop[1]
	yspan = y_start_stop[1]
	# window move
	pixOverX= np.int(window[0]*(1 - overlap[0]))
	pixOverY= np.int(window[1]*(1 - overlap[1]))
	# windows sizes
	nx_buffer = np.int(window[0]*(overlap[0]))
	ny_buffer = np.int(window[1]*(overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/pixOverX)
	ny_windows = np.int((yspan-ny_buffer)/pixOverY)
 
	windowList = []
	for ys in range(ny_windows):
        	for xs in range(nx_windows):
            		startx = xs*pixOverX + x_start_stop[0]
            		endx = startx + window[0]
            		starty = ys*pixOverY + y_start_stop[0]
            		endy = starty + window[1]
			windowList.append(((startx, starty), (endx, endy)))
	return windowList

def drawingBoxes(img, boxes, color=(0, 0, 255), thick=6):
	image = np.copy(img)
	for box in boxes:
        	cv2.rectangle(image, box[0], box[1], color, thick)
	return image



def extractingImgFeatures(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
	imgFeatures = []
 	if color_space != 'RGB':
		featureImg=colorConvert(img,color_space)
	else:
        	featureImg = np.copy(img)

	if spatial_feat == True:
        	sptlFeatures = binSpatial(fearureImg, size=spatial_size)
		imgFeatures.append(sptlFeatures)
 
	if hist_feat == True:
        	histFeatures = colorHist(featureImg, nbins=hist_bins)
	        imgFeatures.append(histFeatures)
  
    	if hog_feat == True:
		if hog_channel == 'ALL':
			hogFeatures = []
			for channel in range(featureImg.shape[2]):
				hogFeatures.append(getHogFeatures(featureImg[:,:,channel],
				orient,pix_per_cell,cell_per_block,
				vis=False))
			hogFeatures=np.ravel(hogFeatures)
		else:
			hogFeatures=getHogFeatures(featureImg[:,:,hog_channel],orient,
					pix_per_cell,cell_per_block,vis=False)
		imgFeatures.append(hogFeatures)

	return np.concatenate(imgFeatures)


def searchWindows(img, windows, cls, X_scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32, orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    	on_windows = []
    	for window in windows:
        	test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        	features = extractingImgFeatures(test_img, color_space=color_space,
                            	spatial_size=spatial_size, hist_bins=hist_bins,
                            	orient=orient, pix_per_cell=pix_per_cell,
                            	cell_per_block=cell_per_block,
                            	hog_channel=hog_channel, spatial_feat=spatial_feat,
                            	hist_feat=hist_feat, hog_feat=hog_feat)
        	X = np.array(features).reshape(1, -1)
        	test_features = X_scaler.transform(X)
        	prediction = cls.predict(test_features)
        	if prediction == 1:
            		on_windows.append(window)
	return on_windows



def findCars(img, ystart, ystop, scale, cls, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
              hog_channel, color_space, spatial_feat, hist_feat, hog_feat):

	on_windows = []
	img = img.astype(np.float32)/255

	img_tosearch = img[ystart:ystop,:,:]
	if  color_space == 'YCrCb':
        	ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    	else:
        	ctrans_tosearch = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


    	if scale != 1:
        	imshape = ctrans_tosearch.shape
		#hata burda!!
        	ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    	ch1 = ctrans_tosearch[:,:,0]
    	ch2 = ctrans_tosearch[:,:,1]
    	ch3 = ctrans_tosearch[:,:,2]

    	# blocks and steps 
    	nxblocks = (ch1.shape[1] // pix_per_cell)-1
    	nyblocks = (ch1.shape[0] // pix_per_cell)-1
    	nfeat_per_block = orient*cell_per_block**2

   
    	window = 64
    	nblocks_per_window = (window // pix_per_cell)-1
    	cells_per_step = 2 
    	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    	hog1 = getHogFeatures(ch1, orient, pix_per_cell, cell_per_block)
    	hog2 = getHogFeatures(ch2, orient, pix_per_cell, cell_per_block)
    	hog3 = getHogFeatures(ch3, orient, pix_per_cell, cell_per_block)

    	for xb in range(nxsteps):
        	for yb in range(nysteps):
        		ypos = yb*cells_per_step
        		xpos = xb*cells_per_step

        	    	if hog_feat:
        	        	if hog_channel == 0:
        	        		hogFeatures = np.array(hog1)[ypos:ypos+nblocks_per_window] [xpos:xpos+nblocks_per_window].ravel()
        	        	elif hog_channel == 1:
        	            		hogFeatures = np.array(hog2)[ypos:ypos+nblocks_per_window] [xpos:xpos+nblocks_per_window].ravel()
        	        	elif hog_channel == 2:
        	            		hogFeatures = np.array(hog3)[ypos:ypos+nblocks_per_window] [xpos:xpos+nblocks_per_window].ravel()
        	        	else:
        	            		hogFeat1 = np.array(hog1)[ypos:ypos+nblocks_per_window] [xpos:xpos+nblocks_per_window].ravel()
        	            		hogFeat2 = np.array(hog2)[ypos:ypos+nblocks_per_window] [xpos:xpos+nblocks_per_window].ravel()
        	            		hogFeat3 = np.array(hog3)[ypos:ypos+nblocks_per_window] [xpos:xpos+nblocks_per_window].ravel()
        	    		        hogFeatures = np.hstack((hogFeat1, hogFeat2, hogFeat3))


        	    		xleft = xpos*pix_per_cell
        	    		ytop = ypos*pix_per_cell

        	    		# Extract the image patch
        	    		subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

        	    		# Get color features
        	    		if spatial_feat:
        	    			sptlFeatures = binSpatial(subimg, size=spatial_size)

        	    		if hist_feat:
        	        		histFeatures = colorHist(subimg, nbins=hist_bins)
	
        	    		X = np.hstack((sptlFeatures, histFeatures, hogFeatures)).reshape(1, -1)
	
        	    		testFeatures = X_scaler.transform(X)
        	    		test_prediction = cls.predict(testFeatures)
	
        	    		if test_prediction == 1:
        	        		xbox_left = np.int(xleft*scale)
        	        		ytop_draw = np.int(ytop*scale)
        	        		win_draw = np.int(window*scale)
        	        		on_windows.append(((xbox_left, ytop_draw+ystart),
							(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
   	return on_window

def addHeat(heatmap, bbox_list):

	    for box in bbox_list:
	        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1 
	    return heatmap


def applyThreshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	return heatmap


def drawBboxes(img, heatmap_buffer, heatmap_pre, N_buffer):

	heatmap_buffer.append(heatmap_pre)
	if len(heatmap_buffer) > N_buffer:
        	heatmap_buffer.pop(0)

	idxs = range(N_buffer)
	for b, w, idx in zip(heatmap_buffer, buffer_weights, idxs):
    	    heatmap_buffer[idx] = b * w

    	heatmap = np.sum(np.array(heatmap_buffer), axis=0)
	threshold= sum(buffer_weights[0:N_buffer])*2
    	heatmap = applyThreshold( heatmap,threshold)

        labels = label(heatmap)

	bboxes = []
	for car_number in range(1, labels[1]+1):
        	nonzero = (labels[0] == car_number).nonzero()
        	nonzeroy = np.array(nonzero[0])
        	nonzerox = np.array(nonzero[1])
        	bbox_tmp = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        	bboxes.append(bbox_tmp)


	for bbox in bboxes:
        	cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)

	return img, heatmap, bboxes

def generateHeatmap(image, windows_list):
	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	heat = add_heat(heat, windows_list)
    	heat = apply_threshold(heat, 1)
    	heatmap = np.clip(heat, 0, 255)
	return heatmap




def vehicleDetect(image):

	start = timer()

	windows_list = []
	for search_window_scale, ystart, ystop in zip(windowScales, ystarts, ystops):
		windows_list_tmp = findCars(np.copy(image), ystart, ystop, windowScales,
 					cls, X_scaler, orient, pix_per_cell, cell_per_block,
                            		spatial_size, hist_bins, hog_channel, color_space, spatial_feat, hist_feat, hog_feat)
        	windows_list.extend(windows_list_tmp)

    	heatmap_pre = generate_heatmap(image, windows_list)

    	draw_img, heatmap_post, bboxes = drawBboxes(np.copy(imgage), copy(Heatmap_buffer), heatmap_pre,
					 min(len(Heatmap_buffer)+1,N_buffer))

	if len(Heatmap_buffer) >= N_buffer:
		Heatmap_buffer.pop(0)
	fps = 1.0 / (timer() - start)


	draw_img = draw_background_highlight(image, draw_img, image.shape[1])
	draw_thumbnails(draw_img, image, bboxes)
	draw_lane_status(draw_img)

	return draw_img


def draw_background_highlight(image, draw_img, w):
	mask = cv2.rectangle(np.copy(image), (0, 0), (w, 155), (0, 0, 0), thickness=cv2.FILLED)
	draw_img = cv2.addWeighted(src1=mask, alpha=0.3, src2=draw_img, beta=0.8, gamma=0)
	return draw_img

def draw_thumbnails(img_cp, img, window_list, thumb_w=100, thumb_h=80, off_x=30, off_y=30):

	cv2.putText(img_cp, 'Detected viehicles', (400,37), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
	for i, bbox in enumerate(window_list):
        	thumbnail = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        	vehicle_thumb = cv2.resize(thumbnail, dsize=(thumb_w, thumb_h))
        	start_x = 300 + (i+1) * off_x + i * thumb_w
        	img_cp[off_y + 30:off_y + thumb_h + 30, start_x:start_x + thumb_w, :] = vehicle_thumb


def draw_lane_status(frame, threshold_offset = 0.6):

	font = cv2.FONT_HERSHEY_SIMPLEX
	info_road = "Lane Status"
	info_lane = "Direction"
	info_cur = "Curvature "
    	info_offset = "Off center:"

    	l_uper = (10,10)

    	cv2.line(frame,(l_uper[0] + 265,0),(l_uper[0] + 265,155),(255,0,0),5)

    	cv2.putText(frame, info_road, (50,32+5), font, 0.8, (255,255,0), 2,cv2.LINE_AA)
    	cv2.putText(frame, info_lane, (16,60+10), font, 0.6, (255,255,0), 1,cv2.LINE_AA)
    	cv2.putText(frame, info_cur, (16,80+25), font, 0.6, (255,255,0), 1,cv2.LINE_AA)

    	if lane_info['offset'] >= threshold_offset:
        	cv2.putText(frame, info_offset, (16,100+40), font, 0.6, (255,0,0), 1,cv2.LINE_AA)
    	else:
        	cv2.putText(frame, info_offset, (16,100+40), font, 0.6, (255,255,0), 1,cv2.LINE_AA)




