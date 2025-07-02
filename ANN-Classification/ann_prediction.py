import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model, scaler pickle , one_hot
model = load_model("model.h5")


# Load the encoder and Scaler
with open('lable_encoder_gender.pkl', 'rb') as file:
    lable_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    lable_encoder_geo = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Input Data
input_data = {
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88,
}

# One Hot Encode 'Geography'
geo_encoded = lable_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=lable_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df

# Convert the key-value pair data into dataFrame
input_data_df = pd.DataFrame([input_data])

# Encode categorical variables (Gender)
input_data_df['Gender'] = lable_encoder_gender.transform(input_data_df['Gender'])

# Concatination one hot encoded data
input_data_df = pd.concat([input_data_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

# Scaling the input data
input_scaled = scaler.transform(input_data_df)

# Predict the churn
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]
print(f"{prediction_probability}")
if(prediction_probability > 0.5) :
    print('The customer is likely to exit')
else :
    print("Then customer is unlikely to exit")





