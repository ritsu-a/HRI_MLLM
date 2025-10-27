

import os
os.environ['MUJOCO_GL'] = 'egl'
from HRI_mllm.external.GMR.scripts.vis_csv_motion import vis_audio_motion
from HRI_mllm.external.HRI_retarget.HRI_retarget.utils.io.motion_pkl_to_csv import load_npz_as_csv_data
import numpy as np


motion_csv = load_npz_as_csv_data("/root/workspace/HRI_MLLM/data/generated_wav_short_3s_with_motion/HAND_WRITE-3/1.npz")
np.savetxt("test1.csv", motion_csv, delimiter=",")



vis_audio_motion("test1.csv", output_path="final_output_2.mp4", audio_path="audio.wav", robot_type="g1_brainco", rate_limit=False)