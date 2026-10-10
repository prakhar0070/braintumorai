# Brain Tumor MRI Image Classification

Deep learning project that classifies brain MRI images into four classes: **glioma, meningioma, no tumor, pituitary**.
It compares a custom CNN with transfer learning models and includes a Streamlit app for predictions.

> Educational project. Not a medical device and not for diagnosis.

## Dataset
- Brain Tumor MRI multi-class dataset (train / valid / test folders, 4 classes).
- Train: 1695 images, Validation: 502, Test: 246.
- Mild class imbalance (glioma has the most images, no tumor the fewest). Class weights were used during training.

## Approach
1. Resize to 224x224, normalise pixels, augment (flips, rotation, zoom, brightness, shifts).
2. Custom CNN (4 conv blocks with BatchNorm, MaxPooling, Dropout).
3. Transfer learning with MobileNetV2 and EfficientNetB0 (ImageNet weights): train the new top layers, then fine-tune the top 30 layers at a low learning rate.
4. EarlyStopping and ModelCheckpoint callbacks.
5. Evaluation on the held-out test set.

## Results (test set)
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Params (M) |
|---|---|---|---|---|---|
| MobileNetV2 | TODO | TODO | TODO | TODO | TODO |
| EfficientNetB0 | TODO | TODO | TODO | TODO | TODO |
| Custom CNN | TODO | TODO | TODO | TODO | TODO |

Confusion matrices and training curves are in the `results/` folder.

## Findings
- TODO: which model is best and why.
- TODO: which classes are confused most often (for example meningioma vs pituitary).
- TODO: custom CNN limitations (it failed to predict some classes).
- The test set is small (246 images), so small differences between models may not be meaningful.

## Run the app
```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Put the trained model at `models/mobilenetv2_final.h5`.

## Project structure
```
app.py                 Streamlit app
requirements.txt
models/                trained models (.h5)
results/               confusion matrices, curves, model_comparison.csv
notebooks/             training notebook
```
