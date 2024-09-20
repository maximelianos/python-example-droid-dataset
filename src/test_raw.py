from pathlib import Path
import cv2

from .raw import RawScene

if __name__ == "__main__":
    scene = Path("data/droid_raw/1.0.1/success/2023-10-27/Fri_Oct_27_19:48:17_2023")
    visualize = False
    raw_scene: RawScene = RawScene(scene, visualize)

    images: dict = raw_scene.log_cameras_next(0)

    plot_path = "data/tmp.jpg"
    cv2.imwrite(plot_path, cv2.cvtColor(images["cameras/ext1/left"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
