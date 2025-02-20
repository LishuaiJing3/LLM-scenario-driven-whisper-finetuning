# awesome-whisper-finetuning-for-low-resource-languages
In this project, I propose a framework to fine tune open source whisper models for transcription of audios by using LLMs. This is very effective for training domain specific whisper models and it can also improve cases where you have a extensive training dataset already. 


Using the Makefile for Awesome Whisper Fine-Tuning

The Makefile provides convenient commands to manage your project for local development and Dockerized deployment.

Local Development
Run APIs Locally

To run the APIs locally using uvicorn with Poetry:

Run Data Curation API:
make data_curation
This starts the Data Curation API at http://localhost:8001.
Run Training API:
make training
This starts the Training API at http://localhost:8002.
Run Serving API:
make serving
This starts the Serving API at http://localhost:8003.
Dockerized Deployment
Build Docker Images



To build Docker images for all services:

make docker-build
This command builds the following Docker images:

data-curation-api for the Data Curation API.
training-api for the Training API.
serving-api for the Serving API.
Run Docker Containers

To start all APIs as Docker containers:

make docker-run
The following services will be available:

Data Curation API: http://localhost:8001
Training API: http://localhost:8002
Serving API: http://localhost:8003
Stop Docker Containers

To stop all running Docker containers for the APIs:

make docker-stop
Clean Docker Environment

To stop and remove all containers and images for the APIs:

make docker-clean
To remove all Docker containers and images (use with caution):

make clean
Additional Information
Environment Variables

Ensure you have an .env file in your project root for configuration. This file should include:

Database connection details.
Model paths.
Other required settings for your APIs.
Swagger Documentation

Each API includes autogenerated Swagger documentation. Once the APIs are running, you can access their documentation at:

Data Curation API: http://localhost:8001/docs
Training API: http://localhost:8002/docs
Serving API: http://localhost:8003/docs
Monitor Logs

To view logs for a running Docker container:

docker logs <container_name>
For example, to check logs for the data-curation-api:

docker logs data-curation-api
Example Workflow
Start the Data Curation API Locally:
make data_curation
Build and Run Dockerized Services:
make docker-build
make docker-run
Stop Services:
make docker-stop
Clean Docker Environment:
make docker-clean



## TODOs
devcontainer for project isolation
CICD
add cloud storage to keep the training data and models so that it does not consume local storage. 
support openAI
DeepSeek models
Token monitoring to LLM
Support more text to speed models to generate audio data for training 
Augmentation of the generated audios: adding background noise, distortion, change pitch, volumns etc.  
Evaluation to the generated audio and transcript
Improve controlling the generated text length 
Test gemini text to speech capability
Control number of records so that we do not run into context window issues. 
pycountry for get country info

Deploy on Kubernettis for hosting
use aisuit to abstract LLM interfaces

Enrich tests
Observability 

add TTS async and parallel capability to speed up training data generation 
add voice cloning capability

Add text and voice alignment capability
Add diversity check for the generated scripts to make audios

Use pubsub or kafka for publish/consume messages when generating scripts and audios.  

# Limitations: 
the text to speed engine you can switch to other models which support more languages. 