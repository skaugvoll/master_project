# Readme
> should change docker-compose to use image: skaugvoll/har_master_app:latest

# Prerequisits
- docker
- nvidia-docker
- cuda
- cudnn
- nvidia graphix card

# IMAGE: to build from scratch (build new image instead of pulling pre-existing)
Currently the images has to be buildt locally using docker-compose
- Thus use this: `docker-compose build`
  - or `nvidia-docker build -t <image_name>`

# RUN container locally (GPU bound)
Since the code is hardcoded to use tensorflow-gpu (and image uses tensorflow-gpu)
we need to install nvidia-docker.
- `nvidia-docker run -d --name <container-name> <image_name | image_id>`
  - E.g: `nvidia-docker run -d --name skaugvoll/har-master-app skaugvoll/har_master_app``


# Building new image and pushing to hub
1. build imag
 - `docker-compose build`
2. tag the image
  - `docker tag docker tag IMAGE_ID <USERNAME_DOCKER_HUB>/<repo_name>`
    - E.g: `docker tag 518a41981a6a skaugvoll/har_master_app`

3. Push to the hub repo
 - `docker push <username>/<repo_name>`
  - E.g: `docker push skaugvoll/har_master_app`


# Jump inside container
`docker exec -it <container_id> bash`

# Run scripts:
**Rembember to use `python3`**

To run inside container; `python3 <script>`
