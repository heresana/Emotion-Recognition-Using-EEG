# utils/label_mapping.py
"""
This file helps to map emotions with each trial
"""
def map_emotion(valence, arousal, threshold=5.0):
    if valence >= threshold and arousal >= threshold:
        return 0  # Happy
    elif valence >= threshold and arousal < threshold:
        return 1  # Calm
    elif valence < threshold and arousal >= threshold:
        return 2  # Angry
    else:
        return 3  # Sad
