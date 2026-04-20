import onnxruntime as ort
import numpy as np
import importlib.resources as pkg_resources

CLASSES = ["Benign", "Mirai", "WannaCry", "Emotet", "Generic"]
def load_session():
    model_path = pkg_resources.files("malvis").joinpath("model.onnx")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

_session = None

def predict(image: np.ndarray) -> tuple[str, float]:
    global _session
    if _session is None:
        _session = load_session()
    inp = image[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,H,W)
    logits = _session.run(["logits"], {"image": inp})[0][0]
    logits -= logits.max()   
    probs = np.exp(logits) / np.exp(logits).sum()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])
