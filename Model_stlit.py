import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and preprocessor
model = joblib.load("model_final_1.pkl")
preprocessor = joblib.load("scaler.pkl")  # If you saved preprocessing steps
reverse_label_mapping = joblib.load("reverse_label_mapping.pkl")

def decode_label(encoded_label):
    return reverse_label_mapping.get(encoded_label, "Unknown Label")

# App title
st.title("Threat Prediction")

# Input form for single prediction
st.header("Enter Feature Values (Single Prediction)")

destination_port = st.number_input("Destination Port", value=0)
flow_duration = st.number_input("Flow Duration", value=0)
total_fwd_packets = st.number_input("Total Forward Packets", value=0)
total_bwd_packets = st.number_input("Total Backward Packets", value=0)
total_length_fwd = st.number_input("Total Length of Forward Packets", value=0.0)
total_length_bwd = st.number_input("Total Length of Backward Packets", value=0.0)
flow_bytes_per_sec = st.number_input("Flow Bytes/s", value=0.0)
flow_packets_per_sec = st.number_input("Flow Packets/s", value=0.0)

# Prediction for single input
if st.button("Predict (Single Entry)"):
    # Combine inputs into a single array
    input_features = np.array([[destination_port, flow_duration, total_fwd_packets,
                                 total_bwd_packets, total_length_fwd, total_length_bwd,
                                 flow_bytes_per_sec, flow_packets_per_sec]])

    # Preprocess inputs if required
    input_features = preprocessor.transform(input_features)

    # Make a prediction
    prediction = model.predict(input_features)

    decoded_prediction = decode_label(prediction[0]) 

    # Display result
    st.success(f"Prediction: {decoded_prediction}")

# Bulk Data Entry Using CSV Upload
st.header("Upload CSV for Bulk Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    
    # Display uploaded data (optional)
    st.write("Uploaded Data:")
    st.write(data)

    # Ensure the required columns exist in the uploaded file
    required_columns = [
        'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s'
    ]
    
    if all(col in data.columns for col in required_columns):
        # Extract the required columns for prediction
        input_features_bulk = data[required_columns].values

        # Preprocess inputs if required
        input_features_bulk = preprocessor.transform(input_features_bulk)

        # Make predictions for the entire batch
        predictions = model.predict(input_features_bulk)

        # Decode predictions
        decoded_predictions = [decode_label(pred) for pred in predictions]

        # Add predictions to the DataFrame
        data['Prediction'] = decoded_predictions

        # Display predictions
        st.write("Predictions for Uploaded Data:")
        st.write(data[['Prediction']])
        
    else:
        st.error(f"Uploaded CSV does not contain all required columns: {', '.join(required_columns)}")

