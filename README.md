# APEM-Project
This project focuses on environmental sound classification for an audio-based medical IoT application, utilizing a low-cost ensemble deep learning model. 
It outlines the preprocessing steps for environmental sound detection through a Python-based co-design approach on the Google Colab platform. The dataset comprises ESC-US, UrbanSound8k, and the IP addresses of devices communicating with the target audio-based medical IoT application. ESC-US and the source IPs are available via the project link, while UrbanSound8k can be accessed using the following code:
import soundata

dataset = soundata.initialize('urbansound8k') dataset.download() # download the dataset dataset.validate() # validate that all the expected files are there

example_clip = dataset.choice_clip() # choose a random example clip print(example_clip) # see the available data
