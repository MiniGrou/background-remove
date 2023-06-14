import io
import os
import typing
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
import moviepy.editor as mpy
import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional
from hsh.library.hash import Hasher
from .u2net import detect, u2net
# from . import utilities
import sys
import requests

import os
import math
import torch.multiprocessing as multiprocessing
import torch
import subprocess as sp
import time
import ffmpeg
import numpy as np
import torch
# from .bg import DEVICE, Net, iter_frames, remove_many
import shlex
import tempfile
import requests
from pathlib import Path

multiprocessing.set_start_method('spawn', force=True)


def worker(worker_nodes,
           worker_index,
           result_dict,
           model_name,
           gpu_batchsize,
           total_frames,
           frames_dict):
    print(F"WORKER {worker_index} ONLINE")

    output_index = worker_index + 1
    base_index = worker_index * gpu_batchsize
    net = Net(model_name)
    script_net = None
    for fi in (list(range(base_index + i * worker_nodes * gpu_batchsize,
                          min(base_index + i * worker_nodes * gpu_batchsize + gpu_batchsize, total_frames)))
               for i in range(math.ceil(total_frames / worker_nodes / gpu_batchsize))):
        if not fi:
            break

        # are we processing frames faster than the frame ripper is saving them?
        last = fi[-1]
        while last not in frames_dict:
            time.sleep(0.1)

        input_frames = [frames_dict[index] for index in fi]
        if script_net is None:
            script_net = torch.jit.trace(net,
                                         torch.as_tensor(np.stack(input_frames), dtype=torch.float32, device=DEVICE))

        result_dict[output_index] = remove_many(input_frames, script_net)

        # clean up the frame buffer
        for fdex in fi:
            del frames_dict[fdex]
        output_index += worker_nodes


def capture_frames(file_path, frames_dict, prefetched_samples, total_frames):
    print(F"WORKER FRAMERIPPER ONLINE")
    for idx, frame in enumerate(iter_frames(file_path)):
        frames_dict[idx] = frame
        while len(frames_dict) > prefetched_samples:
            time.sleep(0.1)
        if idx > total_frames:
            break


def matte_key(output, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit=-1,
              prefetched_batches=4,
              framerate=-1):
    manager = multiprocessing.Manager()

    results_dict = manager.dict()
    frames_dict = manager.dict()


    info = ffmpeg.probe(file_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        file_path
    ]
    framerate_output = sp.check_output(cmd, universal_newlines=True)
    total_frames = int(framerate_output)
    if frame_limit != -1:
        total_frames = min(frame_limit, total_frames)

    fr = info["streams"][0]["r_frame_rate"]

    if framerate == -1:
        print(F"FRAME RATE DETECTED: {fr} (if this looks wrong, override the frame rate)")
        framerate = math.ceil(eval(fr))

    print(F"FRAME RATE: {framerate} TOTAL FRAMES: {total_frames}")

    p = multiprocessing.Process(target=capture_frames,
                                args=(file_path, frames_dict, gpu_batchsize * prefetched_batches, total_frames))
    p.start()

    # note I am deliberately not using pool
    # we can't trust it to run all the threads concurrently (or at all)
    workers = [multiprocessing.Process(target=worker,
                                       args=(worker_nodes, wn, results_dict, model_name, gpu_batchsize, total_frames,
                                             frames_dict))
               for wn in range(worker_nodes)]
    for w in workers:
        w.start()

    command = None
    proc = None
    frame_counter = 0
    for i in range(math.ceil(total_frames / worker_nodes)):
        for wx in range(worker_nodes):

            hash_index = i * worker_nodes + 1 + wx

            while hash_index not in results_dict:
                time.sleep(0.1)

            frames = results_dict[hash_index]
            # dont block access to it anymore
            del results_dict[hash_index]

            for frame in frames:
                if command is None:
                    command = ['ffmpeg',
                               '-y',
                               '-f', 'rawvideo',
                               '-vcodec', 'rawvideo',
                               '-s', F"{frame.shape[1]}x320",
                               '-pix_fmt', 'gray',
                               '-r', F"{framerate}",
                               '-i', '-',
                               '-an',
                               '-vcodec', 'mpeg4',
                               '-b:v', '2000k',
                               '%s' % output]

                    proc = sp.Popen(command, stdin=sp.PIPE)

                proc.stdin.write(frame.tostring())
                frame_counter = frame_counter + 1

                if frame_counter >= total_frames:
                    p.join()
                    for w in workers:
                        w.join()
                    proc.stdin.close()
                    proc.wait()
                    print(F"FINISHED ALL FRAMES ({total_frames})!")
                    return

    p.join()
    for w in workers:
        w.join()
    proc.stdin.close()
    proc.wait()
    return


def transparentgif(output, file_path,
                   worker_nodes,
                   gpu_batchsize,
                   model_name,
                   frame_limit=-1,
                   prefetched_batches=4,
                   framerate=-1):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    cmd = "nice -10 ffmpeg -y -i '%s' -i '%s' -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1,fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -shortest '%s'" % (
        file_path, temp_file, output)
    sp.run(shlex.split(cmd))
    print("Process finished")

    return


def transparentgifwithbackground(output, overlay, file_path,
                      worker_nodes,
                      gpu_batchsize,
                      model_name,
                      frame_limit=-1,
                      prefetched_batches=4,
                      framerate=-1):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Starting alphamerge")
    cmd = "nice -10 ffmpeg -y -i '%s' -i '%s' -i '%s' -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1[fg];[2][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto,fps=10,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' -shortest '%s'" % (
        file_path, temp_file, overlay, output)
    sp.run(shlex.split(cmd))
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return


def transparentvideo(output, file_path,
                     worker_nodes,
                     gpu_batchsize,
                     model_name,
                     frame_limit=-1,
                     prefetched_batches=4,
                     framerate=-1):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Starting alphamerge")
    cmd = "nice -10 ffmpeg -y -nostats -loglevel 0 -i '%s' -i '%s' -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1' -c:v qtrle -shortest '%s'" % (
        file_path, temp_file, output)
    process = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = process.communicate()
    print('after call')

    if stderr:
        return "ERROR: %s" % stderr.decode("utf-8")
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return


def transparentvideoovervideo(output, overlay, file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Starting alphamerge")
    cmd = "nice -10 ffmpeg -y -i '%s' -i '%s' -i '%s' -filter_complex '[1][0]scale2ref[mask][main];[main][mask]alphamerge=shortest=1[vid];[vid][2:v]scale2ref[fg][bg];[bg][fg]overlay=shortest=1[out]' -map [out] -shortest '%s'" % (
        file_path, temp_file, overlay, output)
    sp.run(shlex.split(cmd))
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return


def transparentvideooverimage(output, overlay, file_path,
                         worker_nodes,
                         gpu_batchsize,
                         model_name,
                         frame_limit=-1,
                         prefetched_batches=4,
                         framerate=-1):
    temp_dir = tempfile.TemporaryDirectory()
    tmpdirname = Path(temp_dir.name)
    temp_file = os.path.abspath(os.path.join(tmpdirname, "matte.mp4"))
    matte_key(temp_file, file_path,
              worker_nodes,
              gpu_batchsize,
              model_name,
              frame_limit,
              prefetched_batches,
              framerate)
    print("Scale image")
    temp_image = os.path.abspath("%s/new.jpg" % tmpdirname)
    cmd = "nice -10 ffmpeg -i '%s' -i '%s' -filter_complex 'scale2ref[img][vid];[img]setsar=1;[vid]nullsink' -q:v 2 '%s'" % (
        overlay, file_path, temp_image)
    sp.run(shlex.split(cmd))
    print("Starting alphamerge")
    cmd = "nice -10 ffmpeg -y -i '%s' -i '%s' -i '%s' -filter_complex '[0:v]scale2ref=oh*mdar:ih[bg];[1:v]scale2ref=oh*mdar:ih[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1[out]' -map [out] -shortest '%s'" % (
        temp_image, file_path, temp_file, output)
    sp.run(shlex.split(cmd))
    print("Process finished")
    try:
        temp_dir.cleanup()
    except PermissionError:
        pass
    return

def download_files_from_github(path, model_name):
    if model_name not in ["u2net", "u2net_human_seg", "u2netp"]:
        print("Invalid model name, please use 'u2net' or 'u2net_human_seg' or 'u2netp'")
        return
    print(f"downloading model [{model_name}] to {path} ...")
    urls = []
    if model_name == "u2net":
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2aa',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ab',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ac',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ad']
    elif model_name == "u2net_human_seg":
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2haa',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2hab',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2hac',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2had']
    elif model_name == 'u2netp':
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2netp.pth']
    try:
        os.makedirs(os.path.expanduser("~/.u2net"), exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return

    try:

        with open(path, 'wb') as out_file:
            for i, url in enumerate(urls):
                print(f'downloading part {i+1} of {model_name}')
                part_content = requests.get(url)
                out_file.write(part_content.content)
                print(f'finished downloading part {i+1} of {model_name}')

    finally:
        print()


# closes https://github.com/nadermx/backgroundremover/issues/18
# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda:0')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
# else:
#     DEVICE = torch.device('cpu')

DEVICE = torch.device('cpu')

class Net(torch.nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        hasher = Hasher()
        model = {
            'u2netp': (u2net.U2NETP,
                       'e4f636406ca4e2af789941e7f139ee2e',
                       '1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy',
                       'U2NET_PATH'),
            'u2net': (u2net.U2NET,
                      '09fb4e49b7f785c9f855baf94916840a',
                      '1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
                      'U2NET_PATH'),
            'u2net_human_seg': (u2net.U2NET,
                                '347c3d51b01528e5c6c071e3cff1cb55',
                                '1-Yg0cxgrNhHP-016FPdp902BR-kSsA4P',
                                'U2NET_PATH')
        }[model_name]

        if model_name == "u2netp":
            net = u2net.U2NETP(3, 1)
            path = os.environ.get(
                "U2NETP_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
            ):
                download_files_from_github(
                    path, model_name
                )

        elif model_name == "u2net":
            net = u2net.U2NET(3, 1)
            path = os.environ.get(
                "U2NET_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
                #or hasher.md5(path) != "09fb4e49b7f785c9f855baf94916840a"
            ):
                download_files_from_github(
                    path, model_name
                )

        elif model_name == "u2net_human_seg":
            net = u2net.U2NET(3, 1)
            path = os.environ.get(
                "U2NET_PATH",
                os.path.expanduser(os.path.join("~", ".u2net", model_name + ".pth")),
            )
            if (
                not os.path.exists(path)
                #or hasher.md5(path) != "347c3d51b01528e5c6c071e3cff1cb55"
            ):
                download_files_from_github(
                    path, model_name
                )
        else:
            print("Choose between u2net, u2net_human_seg or u2netp", file=sys.stderr)

        net.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
        net.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
        net.eval()
        self.net = net

    def forward(self, block_input: torch.Tensor):
        image_data = block_input.permute(0, 3, 1, 2)
        original_shape = image_data.shape[2:]
        image_data = torch.nn.functional.interpolate(image_data, (320, 320), mode='bilinear')
        image_data = (image_data / 255 - 0.485) / 0.229
        out = self.net(image_data)[0][:, 0:1]
        ma = torch.max(out)
        mi = torch.min(out)
        out = (out - mi) / (ma - mi) * 255
        out = torch.nn.functional.interpolate(out, original_shape, mode='bilinear')
        out = out[:, 0]
        out = out.to(dtype=torch.uint8, device=torch.device('cpu'), non_blocking=True).detach()
        return out

def download_files_from_github(path, model_name):
    if model_name not in ["u2net", "u2net_human_seg", "u2netp"]:
        print("Invalid model name, please use 'u2net' or 'u2net_human_seg' or 'u2netp'")
        return
    print(f"downloading model [{model_name}] to {path} ...")
    urls = []
    if model_name == "u2net":
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2aa',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ab',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ac',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2ad']
    elif model_name == "u2net_human_seg":
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2haa',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2hab',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2hac',
                'https://github.com/nadermx/backgroundremover/raw/main/models/u2had']
    elif model_name == 'u2netp':
        urls = ['https://github.com/nadermx/backgroundremover/raw/main/models/u2netp.pth']
    try:
        os.makedirs(os.path.expanduser("~/.u2net"), exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        return

    try:

        with open(path, 'wb') as out_file:
            for i, url in enumerate(urls):
                print(f'downloading part {i+1} of {model_name}')
                part_content = requests.get(url)
                out_file.write(part_content.content)
                print(f'finished downloading part {i+1} of {model_name}')

    finally:
        print()


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.asarray(img)
    mask = np.asarray(mask)

    # guess likely foreground/background
    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int64)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)

    return cutout


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout


def get_model(model_name):
    if model_name == "u2netp":
        return detect.load_model(model_name="u2netp")
    if model_name == "u2net_human_seg":
        return detect.load_model(model_name="u2net_human_seg")
    else:
        return detect.load_model(model_name="u2net")


def remove(
    data,
    model_name="u2net",
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_structure_size=10,
    alpha_matting_base_size=1000,
):
    model = get_model(model_name)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    mask = detect.predict(model, np.array(img)).convert("L")

    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    # cutout.seek(0)
    bio.seek(0)

    return bio


def iter_frames(path):
    return mpy.VideoFileClip(path).resize(height=320).iter_frames(dtype="uint8")


@torch.no_grad()
def remove_many(image_data: typing.List[np.array], net: Net):
    image_data = np.stack(image_data)
    image_data = torch.as_tensor(image_data, dtype=torch.float32, device=DEVICE)
    return net(image_data).numpy()
