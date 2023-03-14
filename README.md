# Fashion-Recommender-System

## Problem 
Whenvever we try to shop clothes from any E-Commerce website, we often get overwhelmed by the number of options available to us. This is called as the Paradox of Choice. For example - If we only had to choose between 1% and 2% milk, it is easier to know which option we prefer, since we can easily weigh the pros and cons. When the number of choices increases, so does the difficulty of knowing what is best. 
Instead of increasing our freedom to have what we want, the paradox of choice suggests that having too many choices actually limits our freedom. Therefore, I developed a fashion recommender system.
You upload a picture of item you are looking for and the recommender system will return top 5 similar items from the database.

## Methodology 
For similarity comparison between an image uploaded by the user and the images available in the database, I compared user's image embeddings with the embeddings of each image in the dataset. 
Image embeddings are generated using deep learning model, but training a deep learning model from scratch is very time consuming as well as computionally expensive. Therefore, I utilized transfer learning technique and used ResNet50
that was initilialized by weights obtained from training on ImageNet dataset. I created a sequential Model using ResNet50 and Global MaxPooling layer which takes an image as an 
input and generate a feature vector(image embedding) of size (2048,). To get top 5 recommendations, I applied K-Nearest Neightbor algorithm to get the image embeddings which are closest to 
user's image.

## Model
```
model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
```

## Dataset
[Kaggle - Fashion Product Imge Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

## How to use?

Create a virtual environment and install all the requirements - 
```
pip install -r requirements.txt
```

Run the application -
```
streamlit run main.py
```
