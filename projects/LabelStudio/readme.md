# LabelStudio ML Backend for MMDetection

# To install this repository, clone it and run the following command:
```bash
DOCKER_BUILDKIT=1 docker build -t mmwhale2 --build-context ls_sdk=../label-studio-sdk --build-context ls_ml_backend=../label-studio-ml-backend . 
```

Make sure you have label-studio-sdk and label-studio-ml-backend  repo cloned in the same folder. This will build the image with the label-studio-sdk and label-studio-ml-backend as build contexts.

This repository extends the Label Studio ML backend to support `mmdetection` for detecting eider ducks in aerial imagery.

## Key File
- **`projects/LabelStudio/backend_template/mmdetection_eider_ducks.py`**

---

## Features

### Predict
- **Sliding Window Inference:** Enables sliding window inference directly by adding the sliding window parameter in the config file.

### Fit
1. **Memory Bank for Task Tracking:**
   - Implements a persistent memory bank that tracks previously trained tasks, preventing redundant training.
   - The memory bank is saved as a CACHE (likely using `sqlite3`), ensuring persistence across sessions.

2. **Training on Image Slices:**
   - Supports training on image slices by splitting images into fixed segments using `tools/slice_train_imgs.py`.
   - Slices are saved in the temporary directory and automatically deleted after `fit()` finishes executing.

3. **Keypoint-to-Bounding Box Conversion:**
   - Since original annotations are keypoints, a conversion feature has been added to transform keypoints into bounding boxes.
   - The width and height of the bounding boxes are user-defined via the annotation interface.

4. **Training on Cropped Regions:**
   - Supports training on user-annotated regions by creating cropped image regions with corresponding annotations.
   - These crops are currently saved in the `tmp` directory (persistent for now).

5. **Efficient Memory Handling:**
   - Ensures cache is properly released before and after training to optimize memory usage.

---

## To-Do (Cleanup)
- Move cropped images from feature 4 to the temporary (non-persistent) directory.

---

## Upcoming Features
- Add an API route to force model prediction.
- Implement a feature to force retraining.
- Introduce a reset option to explicitly clear caches and reinitialize the model.
- Enable model checkpoint memory to save the last `n` checkpoints, allowing users to revert to a previous model version if needed.

---

## Known Issues
- Investigate why the `fit()` function is called twice when an annotation is submitted. (Add print statements to examine the triggered event.)

