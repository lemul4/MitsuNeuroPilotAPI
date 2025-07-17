import queue
import threading
import os
import cv2
import json
import numpy as np

def save_data_async(agent, image_depth, image_seg, data):
    idx = agent.data_count
    paths = agent.subfolder_paths
    task = (image_depth, image_seg, data, paths, idx)
    agent._save_queue.put(task)
    agent.data_count += 1

def save_worker(agent):
    while True:
        task = agent._save_queue.get()
        if task is None:
            agent._save_queue.task_done()
            break
        img_depth, img_seg, data, paths, idx = task
        try:
            max_depth_val = np.nanmax(img_depth)
            if max_depth_val > 0:
                depth_vis = (img_depth / max_depth_val * 65535).astype(np.uint16)
            else:
                depth_vis = np.zeros_like(img_depth, dtype=np.uint16)
            cv2.imwrite(os.path.join(paths[0], f"{idx:06d}.png"), depth_vis)

            seg_vis = img_seg
            cv2.imwrite(os.path.join(paths[1], f"{idx:06d}.png"), seg_vis)

            with open(os.path.join(paths[2], f"{idx:06d}.json"), 'w+', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"AutopilotAgent: Error saving task {idx} to {paths}: {e}")
        finally:
            agent._save_queue.task_done()
