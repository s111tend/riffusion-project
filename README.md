# riffusion-project

# Description
With this application you can generate audio with [Riffusion model](https://github.com/riffusion/riffusion) and compose it with your own video. This application uses Gradio for creating UI. You can launch it in Jupyter on your own machine, in GoogleColab or with a Docker container. All the instructions for installation are provided in this file.

# Fast access:
Here are some useful links to use:
- [Google Colab](https://colab.research.google.com/drive/1xrQ0ChXDzc6HmANvCAtClyONL_rOE73X?usp=sharing)
- [DockerHub](https://hub.docker.com/repository/docker/s111tend/riffusion-app/general)

# Installation
## Docker
To run this application with docker you need to pull the last version of the [image](https://hub.docker.com/repository/docker/s111tend/riffusion-app/general) from DockerHub with:
`docker pull s111tend/riffusion-app`
Next you need to run the container:
`docker run -it s111tend/riffusion-app`
After running these commands there will be link to open application.

## GoogleColab
To run this app in GoogleColab, you should open this [link](https://colab.research.google.com/drive/1xrQ0ChXDzc6HmANvCAtClyONL_rOE73X?usp=sharing). Next select at least T4 GPU (it is free for 3 hours). Copy `requirements_colab.txt` file to environment and run all the code cells one by one.

## Your machine
To start app on your machine you need to install requirements from `requirements.txt` file and next run `app.py` file with python or Jupyter Notebook called `Riffusion_test_notebook.ipynb`. You can also inspect the `Dockerfile` and build container with your changes.

### Attention! It is better to run this application on GPU. You can also run it on CPU but it can take some years to load the riffusion model and use it to generate audio lol. Also: to run the docker container you should have about 8GB of available RAM.

# Short user guide
After you open the application you will see simple interface with only one available option: upload video. After uploading your videofile you will able to split it by choosing the number of parts and the part of video you want to work with or trim it by choosing a start and end point. Next you should write prompts for the model: positive prompt is about what music you want to generate (style, type, etc.); negative prompt is about what the model has to avoid while generating spectogram. After these steps you can check if video and audio are correct. Next you can compose it, check the result and download it in .zip format.