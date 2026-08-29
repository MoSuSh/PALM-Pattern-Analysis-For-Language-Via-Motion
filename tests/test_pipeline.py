import numpy as np
from config import SEQUENCE_LENGTH, FEATURE_DIM, TEMPERATURE

def test_feature_dimensions():
    """Verify feature dimensions match system specs (126 features per frame)."""
    assert FEATURE_DIM == 126

def test_input_tensor_shape():
    """Verify temporal buffer tensor shape matches 1D CNN expectations."""
    frame_buffer = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM))
    assert frame_buffer.shape == (15, 126)
    
    # Model batch dimension shape test (1, 15, 126)
    batch_input = np.expand_dims(frame_buffer, axis=0)
    assert batch_input.shape == (1, 15, 126)

def test_temperature_scaling_probabilities():
    """Verify Softmax Temperature Scaling produces valid normalized probabilities."""
    logits = np.array([2.0, 1.0, 0.1])
    scaled_logits = logits / TEMPERATURE
    exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_scaled / np.sum(exp_scaled)
    
    assert np.isclose(np.sum(probs), 1.0)
    assert len(probs) == len(logits)