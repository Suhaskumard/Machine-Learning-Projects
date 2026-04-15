# Crop Yield Prediction Fix - TODO

## Steps to complete:

- [x] **Step 1**: Edit `src/train.py` - Add cleaning of State_Name, Crop, Season (strip + lower) before encoding.
- [x] **Step 2**: Edit `src/predict.py` - Change inputs to lowercase ('karnataka', 'rice', 'kharif').
- [ ] **Step 3**: Retrain model - Run `cd crop-yield-prediction && python src/train.py`
- [ ] **Step 4**: Test prediction - Run `cd crop-yield-prediction && python src/predict.py`
- [ ] **Step 5**: Verify fix (no ValueError, prints predicted yield), update TODO (mark done), cleanup TODO.md

Current progress: Step 3 - Retraining model.
