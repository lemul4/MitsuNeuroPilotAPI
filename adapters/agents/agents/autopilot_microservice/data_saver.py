import os
import threading
import queue
import cv2
import numpy as np
from datetime import datetime
import json

class DataSaver:
    def __init__(self):
        self.save_queue = queue.Queue()
        self._worker_thread = None
        self.initialized_data_saving = False
        self.save_path = None

    def init_save_worker(self, save_path: str):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self._worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._worker_thread.start()
        self.initialized_data_saving = True

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            try:
                self._save_data(item)
            except Exception as e:
                print(f"DataSaver: Error saving data - {e}")
            self.save_queue.task_done()

    def _save_data(self, item):
        now = datetime.now()
        timestamp_str = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

        depth_img = item['image_depth']  # numpy array
        seg_img = item['image_seg']      # numpy array
        data = item['data']              # dict

        # Сохраняем depth image
        depth_path = os.path.join(self.save_path, f"depth_{timestamp_str}.png")
        cv2.imwrite(depth_path, depth_img)

        # Сохраняем segmentation image
        seg_path = os.path.join(self.save_path, f"seg_{timestamp_str}.png")
        cv2.imwrite(seg_path, seg_img)

        # Сохраняем json с метаданными
        data_path = os.path.join(self.save_path, f"data_{timestamp_str}.json")
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)

    def save_data_async(self, image_depth: np.ndarray, image_seg: np.ndarray, data: dict):
        self.save_queue.put({
            'image_depth': image_depth,
            'image_seg': image_seg,
            'data': data,
        })
