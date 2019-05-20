"""Earthquake LSTM model class"""
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers as KL



class EarthquakeLSTM(Model):
    def __init__(self):
        super(EarthquakeLSTM, self).__init__(self)
        

if __name__ == '__main__':
    model = EarthquakeLSTM()
    print('done')
