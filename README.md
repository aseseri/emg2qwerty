# Predicting Keystrokes from Electromyography Signals  

**Abby Seseri**  
**Department of Computer Science**  
**University of California, Los Angeles**  
**abbyseseri@ucla.edu**  

## Abstract  
In this project, I attempted to improve a model that predicts keystrokes from surface electromyography (sEMG) signals using deep learning architectures. The baseline model, which utilized a Time-Depth Separable Convolutional (TDSConv) encoder, was altered to include recurrent architectures such as LSTMs, GRUs, and hybrid CNN-RNN models. I evaluated the performance of these architectures on the emg2qwerty dataset by reducing the Character Error Rate (CER) for a single subject. My experiment included changing model architectures, hyperparameters, and the number of electrode channels. The results suggested that the **TDSLSTMEncoderModified** architecture, with a hidden size of 512 and dropout of 0.2, achieved the best performance, significantly outperforming the baseline and other architectures. This project offers insights into the effectiveness of different architectures for sequential data and potential directions for further work.  

## 1. Introduction  
The **emg2qwerty** dataset allows the decoding of sEMG signals into keystrokes to be studied. The applications of this can greatly benefit a disabled person, allowing the participant to communicate or move using a neural interface. While the baseline TDSConv model has decent performance, I attempted to use recurrent architectures that are better suited for sequential data to improve the accuracy further.  

This project tests the effectiveness of **LSTM, GRU, and hybrid CNN-RNN architectures** in decreasing the CER. Additionally, I experimented with the impact of different architectural changes, such as dropout and layer normalization. My goal is to design deep learning models for sEMG signal decoding that balance performance and computational efficiency. I focused on minimizing CER in a single subject because a personalized model can help reach a higher accuracy for that subject.  

### 1.1 Motivation  
The ability to accurately decode sEMG signals into keystrokes has crucial implications for assistive technologies, particularly for disabled people with motor impairments. Although the baseline TDSConv model has reasonable performance, its reliance on convolutional layers may prevent it from capturing long-term temporal dependencies inherent in sEMG data.  

Recurrent architectures like **LSTMs and GRUs** are suited for sequential data and should improve performance. Furthermore, **hybrid architectures combining convolutional and recurrent layers** could potentially benefit from both spatial and temporal features. By comparing these architectures, this project aims to find the most effective approach for sEMG-based keystroke prediction.  

## 2. Methods  

### 2.1 Dataset and Preprocessing  
The **emg2qwerty** dataset includes sEMG signals recorded from 16 electrode channels on each wrist, sampled at 2 kHz, and respective keystroke labels. For this project, I focused on a single subject (#89335547). The dataset was split into training, validation, and test sets as provided in the baseline code. Inputs were preprocessed using spectrogram normalization and multi-band rotation-invariant MLP transformations.  

### 2.2 Model Architectures  
I replaced the baseline TDSConv encoder with the following architectures:  

#### 2.2.1 TDSLSTMEncoder  
The **TDSLSTMEncoder** is a bidirectional LSTM-based encoder with a fully connected block. LSTMs (Long Short-Term Memory networks) are appropriate for sequential data because they can capture long-term dependencies in time series. The bidirectional nature of the LSTM allows the model to process the input sequence in both forward and backward directions.  

#### 2.2.2 TDSLSTMEncoderModified  
The **TDSLSTMEncoderModified** is an enhanced version of the TDSLSTMEncoder because it includes dropout and layer normalization. These alterations improve generalization ability and training stability.  

#### 2.2.3 TDSGRUEncoder  
The **TDSGRUEncoder** is a bidirectional GRU-based encoder with a fully connected block. GRUs (Gated Recurrent Units) are a variant of LSTMs that use fewer parameters and are computationally more efficient while still capturing temporal dependencies.  

#### 2.2.4 CNNRNNEncoder  
The **CNNRNNEncoder** is a hybrid model that combines 1D convolutional layers for feature extraction and an LSTM for temporal modeling. This approach leverages CNNs for spatial features and LSTMs for sequential dependencies.  

Each architecture retained the **CTC loss function** for training and a decoder for character sequence prediction.  

### 2.3 Training and Evaluation  
- Models were trained using a learning rate scheduler for 30 epochs.  
- Performance was evaluated using Character Error Rate (CER).  
- Various hyperparameters were tested, including hidden sizes, number of layers, and dropout rates.  

## 3. Results  

### 3.1 Performance Comparison  
**Table 1:** CER for different architectures.  

| Architecture          | Test CER | Validation CER | Runtime (minutes) |
|----------------------|----------|----------------|--------------------|
| **Baseline TDSConv** | 28.79    | 28.44          | 54.31              |
| **TDSLSTMEncoder**   | 27.19    | 27.82          | 52.43              |
| **TDSLSTMEncoderModified** | **21.68** | **20.47** | 88.92  |
| **TDSGRUEncoder**    | 28.16    | 27.11          | 52.39              |
| **CNNRNNEncoder**    | 74.86    | 71.13          | 52.42              |

### 3.2 Key Observations  
- **TDSLSTMEncoderModified achieved the lowest CER**, but required nearly double the runtime compared to other models.  
- Recurrent architectures (LSTMs, GRUs) outperformed the baseline TDSConv model.  
- CNNRNNEncoder performed poorly, indicating hybrid architectures may require additional tuning.  
- Increasing hidden size from 64 to 512 consistently improved performance but increased training time.  
- A dropout rate of 0.2 was effective in preventing overfitting.  
- Increasing LSTM layers from 4 to 6 degraded performance, likely due to over-regularization.  

### 3.3 Computational Limitations  
- **Training was limited by Google Colab's GPU resources**.  
- Reducing the number of electrode channels (from 16 to 8) led to worse performance.  

## 4. Discussion  

### 4.1 Insights  
- Temporal modeling is crucial for decoding sEMG signals.  
- Careful hyperparameter tuning is essential (e.g., hidden size, dropout).  
- Hybrid CNN-RNN architectures may require more training time to perform well.  

### 4.2 Limitations  
- High runtime requirements for the best-performing model.  
- Computational constraints prevented testing certain configurations (e.g., 8-channel models).  

### 4.3 Future Work  
- Exploring transformer-based models for sequence prediction.  
- Data augmentation techniques (e.g., time warping, noise injection).  
- Scaling to multiple subjects for improved generalization.  

## References  
- Cha, J., et al. (2021). SWAD: Domain Generalization by Seeking Flat Minima. *NeurIPS*.  
- Graves, A., et al. (2006). Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. *ICML*.  
- Sivakumar, V., et al. (2024). *emg2qwerty: A Large Dataset with Baselines for Touch Typing Using Surface Electromyography.* *arXiv*.  

> *AI-generated code was used as a supplement for my starting point.*  
