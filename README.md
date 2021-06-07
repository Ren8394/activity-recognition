# Activity Recognition Using MLP

This project aims to classify six lower limb activities using 3 axes accelerometers and gyroscopes embedded in wearable sensors. The MLP model of deeplearning is implemented to recognition and classification.

**Lower limb activity**
* right-side straight leg raise (SLR-R)
* left-side straight leg raise (SLR-L)
* right-side short-arc exercise (SAE-R)
* left-side short-arc exercise (SAE-L)
* right-side knee extension (KE-R)
* left-side knee extension (KE-L)

**Sensor**
* **6** OPAL inertial measurement units (published by [APDM](https://apdm.com/wearable-sensors/), Portland, USA)
* **128 Hz** sampling rate
* 6 sensors are attached to the _chest_, _waist_, _both thighs_, and _both shanks_


## Requirement
Tensorflow and Python 3 environment. Using Anoconda to install packages is recommanded.

## Methodology
Execute file in `AR.ipynb` which is a jupyter file
### Load and preprocess data
* Load data
* Labeling
* Normalise
* Split data into training set and testing set **(8:2)**
### Construct neuron network model
* Build Keras MLP
### Validate results
* Confusion Matrix



