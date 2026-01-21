from sam2.build_sam import build_sam2_video_predictor
from aux_functions import *
from Seg2Track import Seg2Track
import argparse
import json
import os
import shutil
import tempfile
from glob import glob

sam2_checkpoint = "/home/james/projects/Seg2Track-SAM2_jr/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


# Parse command line arguments
parser = argparse.ArgumentParser(description="Load config from JSON file")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the JSON configuration file"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    params = json.load(f)
print("Loaded config:", params)

##################################################

subset = params["subset"]  # can be "train" or "testing"
base_path = params["base_path"]
detections_path = params["detections_path"]
params["detection_type"] = params.get("detection_type", "MOT")
batch_size = params.get("batch_size")
batch_id_stride = params.get("batch_id_stride", 10000)


def list_frame_files(frames_dir):
    frame_files = sorted(glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        frame_files = sorted(glob(os.path.join(frames_dir, "*.png")))
    return frame_files


def write_batched_detections(source_path, destination_path, start_idx, end_idx, detection_type):
    with open(source_path, "r") as source_file, open(destination_path, "w") as dest_file:
        for line in source_file:
            if detection_type == "MOT":
                data = line.strip().split(",")
                if len(data) < 7:
                    continue
                frame = int(data[0]) - 1
                if start_idx <= frame < end_idx:
                    data[0] = str(frame - start_idx + 1)
                    dest_file.write(",".join(data) + "\n")
            elif detection_type == "KITTI":
                data = line.strip().split()
                if len(data) < 7:
                    continue
                frame = int(data[0])
                if start_idx <= frame < end_idx:
                    data[0] = str(frame - start_idx)
                    dest_file.write(" ".join(data) + "\n")


def populate_batch_frames(batch_dir, frame_files, dataset_type):
    for idx, frame_path in enumerate(frame_files):
        if dataset_type == "MOT":
            target_name = f"{idx+1:06d}.jpg"
        else:
            target_name = f"{idx:06d}.jpg"
        target_path = os.path.join(batch_dir, target_name)
        try:
            os.symlink(frame_path, target_path)
        except OSError:
            shutil.copy2(frame_path, target_path)

for sequence in os.listdir(f"{base_path}/{subset}"):
    print(f"Sequence {sequence}")
    frames_dir = f"{base_path}/{subset}/{sequence}/img1"
    output_folder = f"{base_path}/output" # Output base directory
    detections_file = detections_path.replace("$sequence$", sequence)

    # Run Seg2Track with reprompting
    device = setup_device()
    predictor = build_sam2_video_predictor(
        model_cfg,
        sam2_checkpoint,
        device=device,
        offload_video_to_cpu=True,
        async_loading_frames=True,
        offload_state_to_cpu=True,
    )

    frame_files = list_frame_files(frames_dir)
    total_frames = len(frame_files)
    if batch_size and batch_size > 0 and total_frames > batch_size:
        for batch_index, start_idx in enumerate(range(0, total_frames, batch_size)):
            end_idx = min(start_idx + batch_size, total_frames)
            batch_frames = frame_files[start_idx:end_idx]
            with tempfile.TemporaryDirectory() as batch_dir:
                populate_batch_frames(batch_dir, batch_frames, params["dataset_type"])
                batch_detection_file = os.path.join(batch_dir, "detections.txt")
                write_batched_detections(
                    detections_file,
                    batch_detection_file,
                    start_idx,
                    end_idx,
                    params["detection_type"],
                )
                inference_state = predictor.init_state(video_path=batch_dir)
                Seg2Track(
                    batch_detection_file,
                    predictor,
                    inference_state,
                    batch_dir,
                    params,
                    output_folder,
                    sequence,
                    frame_offset=start_idx,
                    obj_id_offset=batch_index * batch_id_stride,
                    append_output=batch_index > 0,
                )
                predictor.reset_state(inference_state)
                with torch.no_grad():
                    torch.cuda.empty_cache()
    else:
        inference_state = predictor.init_state(video_path=frames_dir)
        Seg2Track(detections_file, predictor, inference_state, frames_dir, params, output_folder, sequence)
        predictor.reset_state(inference_state)
        with torch.no_grad():
            torch.cuda.empty_cache()
