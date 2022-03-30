import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from utils import *

def video_process(input_path, filename, output_path):
    # Load the DeepLabv3 model to memory
    model = utils.load_model()

    # Start a video cam session
    cap = cv2.VideoCapture(str(input_path) + str(filename))

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    fps = 25

    # Calculate the interval between each frame. 
    interval = int(1000/fps) 
    print("FPS: ",fps, ", interval: ", interval)

    # Read frames from the video, make realtime predictions and display the same
    i = 0
    while True:
        ret, frame = utils.grab_frame(cap)
        if cv2.waitKey(interval) & 0xFF == ord('q'):
                break

        # Ensure there's something in the image (not completely blacnk)
        if np.any(frame) and ret == True:

            if (i % 10 == 0):

                # Read the frame's width, height, channels and get the labels' predictions from utilities
                width, height, channels = frame.shape
                labels = utils.get_pred(frame, model)
                
                # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
                # Hence wherever person is predicted, the label returned will be 15
                # Subsequently repeat the mask across RGB channels 
                mask = (labels != 15)
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
                
                # Resize the image as per the frame capture size
                frame[mask] = 255.0

                mask = (labels == 15)
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
                mask = mask.astype('uint8')
                mask = mask * 255
                
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path) + "/HD/skate_boarder_{}.png".format(str(i)), frame)
                cv2.imwrite(str(output_path) + "/HD/mask/skate_boarder_{}_mask.png".format(str(i)), mask)

                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                c = max(cnts, key = cv2.contourArea)
                x,y,w,h = cv2.boundingRect(c)

                if (w > h):
                    y = max(0, int(y - (w - h) / 2))
                    h = w
                elif (h > w):
                    x = max(0, int(x - (h - w) / 2))
                    w = h
                
                rec = [x, y, w, h]
                frame = frame[y: y + h, x: x + w]
                mask = mask[y: y + h, x: x + w]

                frame = cv2.resize(frame, (512, 512))
                mask = cv2.resize(mask, (512, 512))
                cv2.imwrite(str(output_path) + "/skate_boarder_{}.png".format(str(i)), frame)
                cv2.imwrite(str(output_path) + "/skate_boarder_{}_mask.png".format(str(i)), mask)

                np.savetxt(str(output_path) + "/HD/skate_boarder_{}_rect.txt".format(str(i)), np.array(rec), fmt='%d')        

            i += 1
            
        else:
            break

    # Empty the cache and switch off the interactive mode
    torch.cuda.empty_cache()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Process Video Frames and remove background.')
    parser.add_argument('--output_path', type=str, default="./skate_boarder/", help='Folder where to write the results (save the Barycentric Matrices)')
    parser.add_argument('--input_path', type=str, default="./videos/", help='Path to load the Model, View and Projection Matrices')
    parser.add_argument('--input_filename', type=str, default="Skateboarder.mp4", help='Filename of Video file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[LNDMRKCOR] %(message)s")
    logger = logging.getLogger()

    video_process(args.input_path,
                                            args.input_filename,
                                            args.output_path)