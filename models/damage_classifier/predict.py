import torch
from torchvision import transforms
from PIL import Image
from models.damage_classifier.model import DamageClassifier, CLASSES, IDX_TO_CLASS

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

_model = None


def load_model(path='models/damage_classifier/best_model.pt'):
    global _model
    _model = DamageClassifier.load(path)
    _model.eval()
    print(f'[DL] Damage classifier loaded from {path}')


def predict_damage(image_input):
    """
    image_input: file path string OR PIL.Image object
    Returns dict with severity, severity_idx, confidence, all_probs
    """
    if _model is None:
        raise RuntimeError(
            'Model not loaded. Call load_model() before predict_damage().'
        )

    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    tensor = VAL_TRANSFORMS(img).unsqueeze(0)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = probs.argmax().item()

    return {
        'severity':     IDX_TO_CLASS[pred],
        'severity_idx': pred,
        'confidence':   round(probs[pred].item(), 4),
        'all_probs':    {c: round(probs[i].item(), 4) for i, c in enumerate(CLASSES)}
    }