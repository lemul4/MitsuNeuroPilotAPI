import os
import json
import queue
import threading
import cv2
import numpy as np

class DataSaver:
    def __init__(self):
        self._save_queue = None
        self._worker_thread = None
        self.data_count = 0
        self.subfolder_paths = []
        self.initialized_data_saving = False

    def init_save_worker(self, output_dir):
        import datetime
        os.makedirs(output_dir, exist_ok=True)
        self.subfolder_paths = []
        subfolders = ["depth_front", "instance_segmentation_front", "measurements"]
        for sub in subfolders:
            path = os.path.join(output_dir, sub)
            os.makedirs(path, exist_ok=True)
            self.subfolder_paths.append(path)

        self._save_queue = queue.Queue()
        self._worker_thread = None
        self.initialized_data_saving = True
        print(f"DataSaver: Dataset folders initialized at {output_dir}")

    def _save_worker(self):
        while True:
            task = self._save_queue.get()
            if task is None:
                self._save_queue.task_done()
                break
            img_depth, img_seg, data, paths, idx = task
            try:
                depth_vis_uint16 = img_depth.astype(np.uint16)
                cv2.imwrite(os.path.join(paths[0], f"{idx:06d}.png"), depth_vis_uint16)
                cv2.imwrite(os.path.join(paths[1], f"{idx:06d}.png"), img_seg)
                with open(os.path.join(paths[2], f"{idx:06d}.json"), 'w+', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"DataSaver: Error saving task {idx} to {paths}: {e}")
            finally:
                self._save_queue.task_done()

    def save_data_async(self, image_depth, image_seg, data):
        idx = self.data_count
        paths = self.subfolder_paths
        task = (image_depth, image_seg, data, paths, idx)
        self._save_queue.put(task)
        self.data_count += 1

    def shutdown(self):
        if self._worker_thread and self._worker_thread.is_alive():
            self._save_queue.put(None)
            self._save_queue.join()
            self._worker_thread.join(timeout=5.0)

    @property
    def save_queue(self):
        return self._save_queue

    @property
    def worker_thread(self):
        return self._worker_thread
