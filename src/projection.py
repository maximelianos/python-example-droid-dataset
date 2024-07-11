class CameraStatistic:
    def __init__(self):
        # MV
        # draw trajectory on 2D image
        self.image: np.array = None  # first image with gripper
        self.points = []  # [(y, x, color)]
        self.first_touch = -1  # step number
        self.finger_tip: np.array = np.eye(4)  # [4, 4] link_to_world

        # gripper statistics
        self.is_gripper_closed = False
        self.FPS = 14
        self.gripper_close_count = 0
        self.gripper_duration = 0

        self.visible_points = 0  # is projected 2D point visible

        # compute difference image
        self.first_touch_3d: np.array = None
        self.first_touch_2d: np.array = None
        self.max_distance: float = 0
        self.max_distance_i: int = -1
        self.episode_begin_img: np.array = None
        self.max_distance_img: np.array = None

        # save a few images close to grip moment
        # [ -8, -6, -4, -2, t_grip ]
        # intuition: 1) after grip, the object area doesn't change often
        # 2) after grip, robot region changes before max frame, and object doesn't
        self.max_touch_2d: np.array = None  # [x, y, 1]
        self.last_images: list = []  # (i, frame_i)
        self.final_last_images: list = None  # pointer to images that will be saved
        self.grip_images_before: list = []  # pointer to images near grip moment
        self.grip_images_after: list = []
        self.grip_image: np.array = None