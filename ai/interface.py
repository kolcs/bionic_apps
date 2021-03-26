from abc import ABC, abstractmethod
from datetime import datetime

from numpy import argmax as np_argmax
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from pathlib import Path

TF_LOG = 'tf_log/'


class ClassifierInterface(ABC):

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError


class BaseNet(ClassifierInterface):

    def __init__(self, input_shape, output_shape):
        self._input_shape = input_shape
        self._output_shape = output_shape

        input_tensor, outputs = self._build_graph()
        self._create_model(input_tensor, outputs)

    def _create_model(self, input_tensor, outputs):
        self._model = keras.Model(inputs=input_tensor, outputs=outputs)
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

    def _build_graph(self):
        raise NotImplementedError("Model build function is not implemented")

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y=None, *, validation_data=None, batch_size=None, epochs=1):
        best_model_cp_file = Path(TF_LOG).joinpath('models')
        best_model_cp_file.mkdir(parents=True, exist_ok=True)
        best_model_cp_file = best_model_cp_file.joinpath('best_model.h5')

        self._model.fit(
            x, y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                # TensorBoard(
                #     log_dir=TF_LOG + '/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
                #     update_freq=1
                #     ),
                ModelCheckpoint(
                    str(best_model_cp_file),
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True
                )]
        )

        self._model.load_weights(best_model_cp_file)

    def summary(self):
        self._model.summary()

    def evaluate(self, x, y=None):
        self._model.evaluate(x, y)
