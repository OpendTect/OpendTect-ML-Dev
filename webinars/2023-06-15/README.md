## Webinar 2023-06-15: Building and Training Your Own 2D CNN Model with OpendTect

Here you can download the Jupyter Notebook used in the webinar of June 06, 2023 during first session of dGB Earth Science's back-to-back webinar.

The webinar video can be viewed from these links:
- https://videos.opendtect.org/?id=165
- https://www.youtube.com/watch?v=iWoF0O-qhCw

### Save and Import the Trained Model with Correct Header into OpendTect

```python
#import required library
from dgbpy import mlio as dgbmlio 
from dgbpy import mlapply as dgbmlapply
from dgbpy import keystr as dgbkeys

#load dummy extracted data
imagexfilenm = f'C:/Users/Legion/OdT_Folder/F3_Demo_2023_ML_2/NLAs/2D_extracted_dataset.h5' 
imgdp = dgbmlapply.getScaledTrainingData(imagexfilenm, split=0.2)

#extract required information
infos = imgdp[dgbkeys.infodictstr]

#your trained model
#model = ...

#save model with dgbmlio
dgbmlio.saveModel(model, imagexfilenm, dgbkeys.kerasplfnm, infos, '2D_UNet_VGG19_Real_Webinar.h5')
```


### OpendTect Machine Learning Developers' Community on Discord

<img src="https://dgbes.com/images/discord_logo.svg" width="200px" alt="OpendTect Machine Learning Developers Community" />


- For more information and help join the [OpendTect Machine Learning Developers' Community on Discord](https://discord.gg/9cVrW2sNza)

- For more information on how to become a member and be part of the Community please read the [FAQ](https://dgbes.com/index.php/support/faq-opendtect-machine-learning-developers-community-discord-server)

