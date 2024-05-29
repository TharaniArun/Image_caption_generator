# Image_caption_generator and Speech Generation for Blind Visualization
The goal of image captioning is to convert a given input image into a natural language description.
It involves the concept of computer vision and Natural Language Process to recognize the context of images and describe them in natural language like English.
The caption is then converted into speech for assisting the blind in understanding images.

The task of image captioning can be divided into two modules logically –

-->Image based model : (Extracts the features of our image.
  Usually rely on a Convolutional Neural Network model.)
   
                    
-->Language based model : (which translates the features and objects extracted by our image based model to a natural sentence.Rely on LSTM.)

![image](https://user-images.githubusercontent.com/97394464/215255701-84407ac1-4519-4ba6-98a0-58e020f96d84.png)

A pre-trained CNN extracts the features from our input image. The feature vector is linearly transformed to have the same dimension as the input dimension of LSTM network. This network is trained as a language model on our feature vector.

For training our LSTM model, we predefine our label and target text. For example, if the caption is “An old man is wearing a hat.”, our label and target would be as follows –

Label — [<start> ,An, old, man, is, wearing, a , hat . ]

Target — [ An old man is wearing a hat .,<End> ]

This is done so that our model understands the start and end of our labelled sequence.
  
## DATASET:
  
  Flickr8k_Dataset: Contains 8092 photographs in JPEG format.

Flickr8k_text: Contains a number of files containing different sources of descriptions for the photographs. Flickr_8k_text folder contains file Flickr8k.token which is the main file of our dataset that contains image name and their respective captions separated by newline(“\n”).
  
  The image dataset is divided into 6000 images for training, 1000 images for validation and 1000 images for testing.
  
  - [Flicker8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
- [Flickr_8k_text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

## Files used
- flickr8k_dataset - Dataset folder containing 8091 images.
- flicker8k_text - Dataset folder containing image captions and text files.
- description.txt - Text file containing all image names and their captions after preprocessing
- model.png - Visual representation of the dimensions of our project

- /vgg16/features_vgg.pkl - Pickle object containing an image and its feature vector from the VGG16 pre-trained CNN model
- /vgg16/best_model.h5 -  H5 file which contains our trained model

- /xception/features_vgg.pkl Pickle an object containing an image and its feature vector from the XCeption pre-trained CNN model
- /xception/best_model.h5 -  H5 file which contains our trained model

You can download all the files from the below drive link

[Drive link](https://drive.google.com/drive/folders/1jC2lha4tKoD6UI9MKCKYAEisCsHs9fHj?usp=share_link)

  
  ## Model Architecture
The model architecture consists of a CNN(VGG16 or XCeption) that extracts the features and encodes the input image and a RNN based on LSTM layers. The most significant difference with other models is that the image embedding is provided as the first input to the RNN network and only once.
- We remove the last layer of VGG or XCeption network
- Image is fed into this modified network to generate a 2048 length encoding corresponding to it
- The 2048 length vector is then fed into a second neural network along with a caption for the image (while training)
- This second network consists of an LSTM which tries to generate a caption for the image
  
 ![image](https://user-images.githubusercontent.com/97394464/215256097-d4825aa1-aa34-48f1-93ea-91d09551f480.png)
  
--Preprocessing of Image

--Creating the vocabulary for the image

--Train the set

--Evaluating the model

--Testing on individual images
  
  ## Results
### Using VGG16 model:
- BLEU-1:  58%
- BLEU-2:  34%
- BLEU-3:  25%
- BLEU-4:  13%


### Using XCeption model:
- BLEU-1:  72%
- BLEU-2:  36%
- BLEU-3:  25%
- BLEU-4:  12%
  
  ![image](https://user-images.githubusercontent.com/97394464/215256595-c82408ba-165d-4c2b-8cc8-7ed08b94f59c.png)

  
 # Example:
  ![image](https://user-images.githubusercontent.com/97394464/215256661-63ae0091-58df-4055-9f40-b5875c88b10d.png)

  
 ## Conclusion:
  
We build a deep learning model with the help of CNN and LSTM. We used a very small dataset of 8000 images to train our model, but the business level model used larger datasets of more than 100,000 images for better accuracy. The larger the datasets are higher the accuracy. So, if you want to build a more accurate caption generator you can try this model with large datasets.
  
  


  
