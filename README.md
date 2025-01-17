# awesome-whisper-finetuning-for-low-resource-languages
In this project, I propose a framework to fine tune open source whisper models for transcription of audios by using LLMs. This is very effective for training domain specific whisper models and it can also improve cases where you have a extensive training dataset already. 


## TODOs
devcontainer for project isolation
CICD
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

Deploy on Kubernettis
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