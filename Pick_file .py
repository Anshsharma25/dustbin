import pickle

# Define metadata for garbage detection model
garbage_model_metadata = {
    "model_path": "garbagev1.pt",
}

# Define metadata for dry/wet classification model
drywet_model_metadata = {
    "model_path": "Drybest.pt",
}

# Combine both models' metadata
all_models_metadata = {
    "garbage_model": garbage_model_metadata,
    "drywet_model": drywet_model_metadata
}

# Save to a pickle file
pickle_file_path = "models_metadata.pkl"
with open(pickle_file_path, "wb") as file:
    pickle.dump(all_models_metadata, file)

print(f"Metadata saved successfully to {pickle_file_path}")
