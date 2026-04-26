"""
Models registry for the MATLAB pipeline.
Re-exports the seven detector architectures from step3_multiModel,
parameterised for n_features=9 (MATLAB 9-d state vector).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'step3_multiModel'))

from model import (
    GPSSpoofingDetector  as CNN,
    LSTMDetector         as LSTM,
    BiLSTMDetector       as BiLSTM,
    GRUDetector          as GRU,
    CNNLSTMDetector      as CNNLSTM,
    TCNDetector          as TCN,
    TransformerDetector  as Transformer,
)

# Registry: name → class
MODEL_REGISTRY = {
    'CNN':        CNN,
    'LSTM':       LSTM,
    'BiLSTM':     BiLSTM,
    'GRU':        GRU,
    'CNN-LSTM':   CNNLSTM,
    'TCN':        TCN,
    'Transformer': Transformer,
}

__all__ = ['MODEL_REGISTRY', 'CNN', 'LSTM', 'BiLSTM', 'GRU', 'CNNLSTM', 'TCN', 'Transformer']
