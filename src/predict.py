# --- 5. Prediction Function for New Data ---
def predict_new_house_price(model, new_data_dict, training_columns):
    # Create a DataFrame from the new data
    new_house_df = pd.DataFrame([new_data_dict])

    # Reindex to match the columns of the training data
    final_df = new_house_df.reindex(columns=training_columns, fill_value=0)

    # Make the prediction
    predicted_price = model.predict(final_df)[0]

    return predicted_price

# Example of how to use the prediction function
new_house_features = {
    'Rooms': 3,
    'Bathroom': 2,
    'Car': 2,
    'Landsize': 500,
    'Distance': 6.5,
    'Type_t': 0, # Not a townhouse
    'Type_u': 0, # Not a unit (so it's a house)
    'Regionname_Northern Metropolitan': 1,
    'Regionname_Southern Metropolitan': 0,
    'Regionname_Western Metropolitan': 0
}

# The other one-hot encoded columns would be 0 by default, so we don't need to list them all.
predicted_price = predict_new_house_price(model, new_house_features, training_columns)

print(f"Prediction for a new house:")
print(f"Features: {new_house_features}")
print(f"Predicted Price: {predicted_price:,.0f} AUD")
