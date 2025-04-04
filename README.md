# riffusion-project

# Description
With this application you can generate audio with [Riffusion model](https://github.com/riffusion/riffusion) and compose it with your own video. This application uses Gradio for creating UI. You can launch it in Jupyter on your own machine, in GoogleColab or with a Docker container. All the instructions for installation are provided in this file.

# Fast access:
Here are some useful links to use:
- [Google Colab](https://colab.research.google.com/drive/1xrQ0ChXDzc6HmANvCAtClyONL_rOE73X?usp=sharing) (Recommended)

# Installation

## GoogleColab
To run this app in GoogleColab, you should open this [link](https://colab.research.google.com/drive/1xrQ0ChXDzc6HmANvCAtClyONL_rOE73X?usp=sharing). Next select at least T4 GPU (it is free for 3 hours). Copy `requirements_colab.txt` file to environment and run all the code cells one by one.

### Attention! It is better to run this application on GPU. You can also run it on CPU but it can take some years to load the riffusion model and use it to generate audio lol.

# Short user guide
After you open the application you will see simple interface with only one available option: upload video. After uploading your videofile you will able to split it by choosing the number of parts (maximum 6 because I didn't find good method to make better optimized output) and the part of video you want to work with or trim it by choosing a start and end point. Next you should write prompts for the model: positive prompt is about what music you want to generate (style, type, etc.); negative prompt is about what the model has to avoid while generating spectogram. After these steps you can check if video and audio are correct. Next you can compose it, check the result and download it in .zip format.