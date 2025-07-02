import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard



# Step 1: Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

# Step 2: Preprocessing

# Remove unnecessary data
data = data.drop(['RowNumber', "CustomerId", "Surname"], axis=1) 

# Encode lable variable. Eg. Gender
lable_encoder_gender = LabelEncoder()
data['Gender'] = lable_encoder_gender.fit_transform(data['Gender'])

# convert the category data feature into numerical feature ('Geograhy' column) using one hot encode 
one_hot_encoder_geo = OneHotEncoder()
geo_encoder = one_hot_encoder_geo.fit_transform(data[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoder.toarray(), columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one hot encoder columns with the original data
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

## Step 3: Save the encoder and scaler
with open('lable_encoder_gender.pkl', 'wb') as file:
    pickle.dump(lable_encoder_gender, file)

with open('one_hot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(one_hot_encoder_geo, file)

# Divide the dataset into independent and dependent features
X = data.drop('Exited', axis=1)
Y = data['Exited']

# Split the data in training and testing test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# You split the data into:
# X_train, Y_train → 80% for training the model
# X_test, Y_test → 20% for testing (evaluating) the model
# test_size=0.2 means 20% goes to testing.
# random_state=42 ensures that every time you run this, the split is the same (for reproducibility).

# Scale these features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('scalar.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Build our ANN Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1], )), # HL1 Connected with input Layer
    Dense(32, activation='relu'), # HL2
    Dense(1, activation='sigmoid') # output Layer
])

# Compile the model

# Use optimizer
opt= tensorflow.keras.optimizers.Adam(learning_rate=0.01)
# Use some loss
loss = tensorflow.keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics = ['accuracy'])

# Set up the Tesnorboard
log_dir = "log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback = TensorBoard(log_dir= log_dir, histogram_freq=1)

# Set up Early Stopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights=True)


# Training the Model
history = model.fit(
    X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100,
    callbacks = [tensorflow_callback, early_stopping_callback]
)
model.save('model.h5')






