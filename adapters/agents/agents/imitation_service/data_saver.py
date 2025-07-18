import os
import threading
import queue
import json
import numpy as np
import cv2

class DataSaver:
    """
    Асинхронное сохранение данных (изображения и json).
    """
    def __init__(self, base_save_path):
        self.base_save_path = base_save_path
        self.subfolder_paths = []
        self._save_queue = queue.Queue()
        self.data_count = 0
        self._worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._worker_thread.start()
        self._init_folders()

    def _init_folders(self):
        os.makedirs(self.base_save_path, exist_ok=True)
        subfolders = [
            "depth_front", "instance_segmentation_front", "measurements"
        ]
        for sub in subfolders:
            path = os.path.join(self.base_save_path, sub)
            os.makedirs(path, exist_ok=True)
            self.subfolder_paths.append(path)
        print(f"DataSaver: Dataset folders initialized at {self.base_save_path}")

    def save_async(self, image_depth, image_seg, data):
        idx = self.data_count
        task = (image_depth, image_seg, data, self.subfolder_paths, idx)
        self._save_queue.put(task)
        self.data_count += 1

    def _save_worker(self):
        while True:
            task = self._save_queue.get()
            if task is None:
                self._save_queue.task_done()
                break
            img_depth, img_seg, data, paths, idx = task
            try:
                max_depth_val = np.nanmax(img_depth)
                if max_depth_val > 0:
                    depth_vis = (img_depth / max_depth_val * 65535).astype(np.uint16)
                else:
                    depth_vis = np.zeros_like(img_depth, dtype=np.uint16)
                cv2.imwrite(os.path.join(paths[0], f"{idx:06d}.png"), depth_vis)

                cv2.imwrite(os.path.join(paths[1], f"{idx:06d}.png"), img_seg)

                with open(os.path.join(paths[2], f"{idx:06d}.json"), 'w+', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"DataSaver: Error saving task {idx} to {paths}: {e}")
            finally:
                self._save_queue.task_done()

    def shutdown(self):
        self._save_queue.put(None)
        self._save_queue.join()
        self._worker_thread.join(timeout=5.0)
        if self._worker_thread.is_alive():
            print("DataSaver: Warning - Save worker thread did not terminate cleanly.")
        else:
            print("DataSaver: Save worker thread terminated.")
