import re

def clean_string(value):
    value = value.lower() 
    value = re.sub(r'[^a-z0-9]', '_', value) 
    value = re.sub(r'_+', '_', value) 
    value = value.strip('_')
    return value

def get_columns():

    columns = [
        "artist",
        "title",
        "genre",
        "tempo",
        "rms_mean",
        "rms_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_flatness_mean",
        "spectral_flatness_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zcr_mean",
        "zcr_std",
        "onset_strength_mean",
        "onset_strength_std",
        "dynamic_range_db"
    ] 

    chroma_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    for note in chroma_notes:
        columns.append(f"chroma_{note}_mean")

    for note in chroma_notes:
        columns.append(f"chroma_{note}_std")

    # 20 MFCC coefficients
    for i in range(1, 21):
        columns.append(f"mfcc_{i}_mean")

    for i in range(1, 21):
        columns.append(f"mfcc_{i}_std")

    # 7 spectral contrast bands
    for i in range(1, 8):
        columns.append(f"spectral_contrast_band{i}_mean")
    
    for i in range(1, 8):
        columns.append(f"spectral_contrast_band{i}_std")

    # 6 tonnetz dimensions
    for i in range(1, 7):
        columns.append(f"tonnetz_{i}_mean")
    
    for i in range(1, 7):
        columns.append(f"tonnetz_{i}_std")

    # Tempogram ratio factors (adjust count as needed)
    for i in range(1, 14):
        columns.append(f"tempogram_ratio_factor{i}_mean")
    
    for i in range(1, 14):
        columns.append(f"tempogram_ratio_factor{i}_std")

    return columns