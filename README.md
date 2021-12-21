# Floor-Level-Net
- Code for paper "FloorLevel-Net: Recognizing Floor-Level Lines With Height-Attention-Guided Multi-Task Learning."
(which is mainly based on the code of "Cars Can’t Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks" Many thanks to the [authors](https://github.com/shachoi/HANet)).
- Pretrained model and example datasets under our augmentation scheme can be downloaded with [link](https://drive.google.com/drive/folders/1L4-zP1hC9F8ZekkO5-oqcRDVqO56BmrI?usp=sharing).
- Dataset for floor recognition of facades and augmented code are pending release.
- For training your own model, please download the augmented dataset and put them into '/floor_data'. Also, the pretrained res101 model should be put in the '/pretrained' folder. Use 'python train.py' for training.
- For inference your own collected images, please download the pre-trained models and put them into '/models', and uncommented the 'inference()' function. You should put the images into the '/floor_data/real_data' folder, and results will be put into the same folder after running.