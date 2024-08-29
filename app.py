import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load the model and unique brand values
model = joblib.load('model.joblib')
unique_values = joblib.load('unique_values.joblib')
name_values = unique_values['name']
fuel_values = unique_values['fuel']
seller_type_values = unique_values['seller_type']
transmission_values =unique_values['transmission']
owner_values = unique_values['owner']

# Define the prediction function
def predict(name, year, km_driven, fuel, seller_type, transmission, owner):
    # Convert inputs to appropriate types
    year = int(year)
    km_driven = int(km_driven)

    # Prepare the input array for prediction
    input_data = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel' : [fuel],
        'seller_type' : [seller_type],
        'transmission' : [transmission],
        'owner' : [owner]
    })

    # Perform the prediction
    prediction = model.predict(input_data)

    return prediction[0]

    # Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=list(name_values), label="Name"),
        gr.Textbox(label="Year"),
        gr.Textbox(label="Kilometer Driven"),
        gr.Dropdown(choices=list(fuel_values), label="Fuel"),
        gr.Dropdown(choices=list(seller_type_values), label="Seller Type"),
        gr.Dropdown(choices=list(transmission_values), label="Transmission"),
        gr.Dropdown(choices=list(owner_values), label="Owner")
    ],
    outputs="text",
    title="Sell Price Predictor",
    description="Enter the Name, Year, Kilometer Driven, Fuel, Seller Type, Transmission and Owner to predict the target value."
)

# Launch the app
interface.launch()