import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append('riffusion-inference')

import os
import sys
import numpy as np
import gradio as gr
import torch

from diffusers import DiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from io import BytesIO
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip #is faster than subclip function
from zipfile import ZipFile

# RIFFUSION PIPELINE
pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1") #pipeline loading
if torch.cuda.is_available():
    pipe = pipe.to("cuda") #switching device to cude if it is available

params = SpectrogramParams()
converter = SpectrogramImageConverter(params=params)

# MAIN FUNCTIONS
def predict(prompt, negative_prompt, width, output_dir = ''):
    """
        Function returns pipeline output converted to audiofile
    """
    spec = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
    ).images[0]
    wav = converter.audio_from_spectrogram_image(image=spec)
    output_file = output_dir + '/output.wav'
    wav.export(output_file, format='wav')
    return output_file

def split_clip(videofile: str, trim_by_parts: bool, parts = 2, start_point = 0, end_point = None):
    """
        Function splits video and returns files' names
    """
    clips = []
    split = videofile.split('/')
    filename, codec = split[-1].split('.')
    path = '/'.join(split[:-1])
    video = VideoFileClip(videofile)
    if trim_by_parts:
        one_subclip_duration = video.duration / parts
        for part in range(parts - 1):
            clips.append(f'{path}/{filename}_{part}.{codec}')
            ffmpeg_extract_subclip(videofile, part*one_subclip_duration, (part + 1)*one_subclip_duration, targetname=clips[-1])
        clips.append(f'{path}/{filename}_{parts - 1}.{codec}')
        ffmpeg_extract_subclip(videofile, (parts - 1)*one_subclip_duration, video.duration, targetname=clips[-1])
    else:
        time_gates = []
        if start_point > 0:
            time_gates.append((0, start_point))
        time_gates.append((start_point, end_point))
        if end_point < video.duration:
            time_gates.append((end_point, video.duration))
        for part, (start, end) in enumerate(time_gates):
            clips.append(f'{path}/{filename}_{part}.{codec}')
            ffmpeg_extract_subclip(videofile, start, end, targetname=clips[-1])
    return clips

def add_audio(videofile: str, audiofile: str, overlay_audio):
    """
        Function composes audio and video and returns resulting file
    """
    video = VideoFileClip(videofile)
    audio = AudioFileClip(audiofile)
    if audio.duration < video.duration:
        repeats = int(np.ceil(video.duration/audio.duration))
        audio_clips = [audio] * repeats
        audio = concatenate_audioclips(audio_clips)
    audio = audio.subclip(0, video.duration)
    if overlay_audio and video.audio:
        video = video.set_audio(CompositeAudioClip([audio, video.audio]))
    else:
        video.audio = audio
    filename, codec = videofile.split('.')
    videofile = filename + '_modified.' + codec
    video.write_videofile(videofile)
    return videofile

# GUI FUNCTIONS
def on_video_upload(video):
    length = VideoFileClip(video).duration
    return (length,
            gr.Number(value=0, minimum=0, maximum=length - 1, label='From (s): ', interactive=True),
            gr.Number(value=length, minimum=1, maximum=length, label='To (s):', interactive=True))

def on_parts_number_change(parts):
    return gr.Number(value=1, minimum=1, maximum=parts, label='Part to work with: ', interactive=True)

def on_start_point_change(start_point, video_duration):
    return gr.Number(value=video_duration, minimum=start_point + 1, maximum=video_duration, label='To (s):', interactive=True)

def on_split_button_click(video, parts, clip_id):
    try:
        clips = split_clip(video, trim_by_parts=True, parts=parts)
        return clips[clip_id - 1], VideoFileClip(clips[clip_id - 1]).duration, clips, clip_id, gr.Button('Generate', interactive=True)
    except Exception:
        return None, None, None, None, gr.Button('Generate', interactive=False)

def on_trim_button_click(video, start_point, end_point):
    clip_id = 1 if start_point == 0 else 2
    clips = split_clip(video, trim_by_parts=False, start_point=start_point, end_point=end_point)
    return clips[clip_id - 1], VideoFileClip(clips[clip_id - 1]).duration, clips, clip_id, gr.Button('Generate', interactive=True)

def on_generate_button_click(positive_prompt, negative_prompt, duration):
    audio_width = int(np.ceil(duration*100.2/8)*8)
    return predict(positive_prompt, negative_prompt, audio_width), gr.Button('Compose', interactive=True)

def on_compose_button_click(video, audio, overlay_audi, clips, clip_number):
    try:
        composed_video = add_audio(video, audio, overlay_audio)
        clips[clip_number - 1] = composed_video
        resulting_videos = [gr.Video(clip, visible=True, interactive=False) for clip in clips]
        return clips, gr.DownloadButton('Download (.zip)', interactive=True), gr.Button('Compose', interactive=True)
    except Exception:
        return None, gr.DownloadButton('Download (.zip)', interactive=False), gr.Button('Compose', interactive=False)

def on_res_clips_change(resulting_clips):
    if not resulting_clips:
        results = [gr.Video(label='Results', interactive=False, value=None)] + [gr.Video(visible=False) for _ in range(5)]
    else:
        results = [gr.Video(clip, visible=True, label=clip.split('/')[-1], interactive=False, include_audio=True) for clip in resulting_clips]
        while len(results) < 6:
            results.append(gr.Video(visible=False))

    zip_filename = "clips.zip"
    with ZipFile(zip_filename, 'w') as zipf:
        for videofile in resulting_clips:
            zipf.write(videofile, os.path.basename(videofile))

    return [zip_filename] + results

# INTERFACE
with gr.Blocks() as demo:
    gr.Markdown('''
	# App that can help you to compose your video with generated audio :)
	Upload your video -> Split it or trim -> Generate audio with riffusion model -> Check if everything is ok -> Compose video and audio -> Download it
    ''')
    # Main user inputs:
    with gr.Row(equal_height=True):
        video = gr.Video(label='Upload your videofile: ')
        video_duration = gr.Number(visible=False, value=10, minimum=0)
        with gr.Tab('Split'):
            parts = gr.Number(value=1, minimum=1, maximum=6, label='Parts to split (Max 6): ')
            clip_number = gr.Number(label='Part to work with: ', value=1, maximum=1)
            split_btn = gr.Button('Split')
        with gr.Tab('Trim'):
            start_point = gr.Number(value=0, minimum=0, label='From (s): ')
            end_point = gr.Number(value=1, minimum=1, label='To (s):', interactive=True)
            trim_btn = gr.Button('Trim')

        clip_duration = gr.Number(visible=False, value=10, minimum=0)

        with gr.Column():
            positive_prompt = gr.Textbox(lines=2, label='Positive prompt: ')
            negative_prompt = gr.Textbox(lines=2, label='Negative prompt: ')

            generate_btn = gr.Button('Generate', interactive=False)

    clips = gr.State([])
    clip_id = gr.State([])

    # Validation widgets for the user:
    with gr.Column():
        with gr.Row(equal_height=True):
            videoclip = gr.Video(label='Target clip: ', interactive=False)
            with gr.Column():
                generated_audio = gr.Audio(label='Generated audio: ', type='filepath', interactive=False)
                overlay_audio = gr.Checkbox(label="Overlay audio", info="")
                clear_video_btn = gr.ClearButton(value='Unset target videoclip', components=[videoclip], )
                clear_audio_btn = gr.ClearButton(value='Unset generated audio', components=[generated_audio])

        compose_btn = gr.Button('Compose', interactive=False)

    progress = gr.Progress()
    resulting_clips = gr.State([])

    # Outputs:
    # TODO: make another output without maximum=6 clips (videogallery doesn't work)
    with gr.Column():
        with gr.Row():
            res_v1, res_v2, res_v3 = gr.Video(label='Results'), gr.Video(visible=False), gr.Video(visible=False)
        with gr.Row():
            res_v4, res_v5, res_v6 = gr.Video(visible=False), gr.Video(visible=False), gr.Video(visible=False)

    # Download and clear buttons:
    with gr.Row():
        zip_btn = gr.DownloadButton('Download (.zip)', interactive=False)
        clear_all_btn = gr.ClearButton(
            components=[video, positive_prompt, negative_prompt, videoclip, generated_audio, resulting_clips],
            value='Clear all'
        )

    # EVENT CATCHERS
    video.upload(
        fn=on_video_upload, 
        inputs=video, 
        outputs=[video_duration, start_point, end_point]
    )
    parts.change(
        fn=on_parts_number_change, 
        inputs=parts, 
        outputs=clip_number
    )
    start_point.change(
        fn=on_start_point_change, 
        inputs=[start_point, video_duration], 
        outputs=end_point
    )
    split_btn.click(
        fn=on_split_button_click, 
        inputs=[video, parts, clip_number], 
        outputs=[videoclip, clip_duration, clips, clip_id, generate_btn]
    )
    trim_btn.click(
        fn=on_trim_button_click, 
        inputs=[video, start_point, end_point], 
        outputs=[videoclip, clip_duration, clips, clip_id, generate_btn]
    )
    generate_btn.click(
        fn=on_generate_button_click, 
        inputs=[positive_prompt, negative_prompt, clip_duration], 
        outputs=[generated_audio, compose_btn]
    )
    compose_btn.click(
        fn=lambda x: gr.Button('Composing video and audio... Please wait', interactive=False), 
        outputs=compose_btn
    ).then(
        fn=on_compose_button_click,
        inputs=[videoclip, generated_audio, overlay_audio, clips, clip_id],
        outputs=[resulting_clips, zip_btn, compose_btn]
    )
    resulting_clips.change(
        fn=on_res_clips_change, 
        inputs=resulting_clips, 
        outputs=[zip_btn, res_v1, res_v2, res_v3, res_v4, res_v5, res_v6]
    )

# Launching demo
demo.launch(debug=False, share=True)