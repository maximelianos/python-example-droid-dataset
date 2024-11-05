import numpy as np

class ImageSaver:
    cache: list = [] # [(time, image)]
    t_max_diff: int = 1000  # millis, max queue time diff
    t_step: int = 100       # millis, min time diff btw close frames
    center_images: list = [] # [(time, image)]
    center_time: int = 0    # millis, save some frames before and after
    snapshots: dict = {}    # single images {description: image}

    def __init__(self):
        pass

    def append(self, time: int, image: np.array):
        # discard frame if too little time passed from previous frame
        if len(self.cache) > 0 and abs(time - self.cache[-1][0]) < self.t_step:
            return

        self.cache.append((time, image))

        # pop from begin if time difference > t_max_diff
        if time - self.cache[0][0] > self.t_max_diff:
            self.cache.pop(0)

        # save images after the Moment
        if abs(time - self.center_time) < self.t_max_diff:
            self.center_images.append((time, image))

    def save_center(self, center_time: int):
        self.center_time = center_time
        for time, image in self.cache:
            # save images before the Moment
            if abs(time - self.center_time) < self.t_max_diff:
                self.center_images.append((time, image))

    def snap(self, desc: str, image: np.array, replace=False):
        if not desc in self.snapshots or replace:
            self.snapshots[desc] = image