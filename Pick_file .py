import pickle

# Define metadata for garbage detection model
garbage_model_metadata = {
    "model_path": "garbagev1.pt",
}

# Define metadata for dry/wet classification model
drywet_model_metadata = {
    "model_path": "Main_DW.pt",
}

# Define metadata for cover/uncover classification model
cover_uncover_model_metadata = {
    "model_path": "cover_model.pt",
}

# Define metadata for polythene/non-polythene classification model
polythene_nonpoly_model_metadata = {
    "model_path": "poly_non_poly.pt",
}

# Combine all models' metadata
all_models_metadata = {
    "garbage_model": garbage_model_metadata,
    "drywet_model": drywet_model_metadata,
    "cover_uncover_model": cover_uncover_model_metadata,
    "polythene_nonpoly_model": polythene_nonpoly_model_metadata
}

# Save to a pickle file
pickle_file_path = "models_pickle.pkl"
with open(pickle_file_path, "wb") as file:
    pickle.dump(all_models_metadata, file)

print(f"Metadata saved successfully to {pickle_file_path}")