import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

class VideoToFrames:
    def __init__(self, video_path, no_frames=30):
        self.reader = cv2.VideoCapture(video_path)

        self.no_frames = no_frames
        self.total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames != 0:
            self.interval = int(self.total_frames/self.no_frames)

    def convert(self):
        frames = 0
        frame_number = 1
        while frames < self.no_frames:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.reader.read()

            try:
                if frame == None:
                    break
            except ValueError:
                pass

            frames += 1
            frame_number += self.interval
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        return frames



if __name__ == '__main__':
    dir = 'videoFolder/'

    no_frames = []
    files = []

    for file in tqdm(os.listdir(dir)):
        path = os.path.join(dir, file)
        converter = VideoToFrames(path)
        no_frames.append(converter.convert())
        files.append(file)

    df = pd.DataFrame([files, no_frames])
    df = df.transpose()
    df.columns = ['file', 'noFrames']
    print(df.describe())
    df.to_csv('File-Frames.csv')
