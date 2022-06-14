from abc import ABC, abstractmethod
from pathlib import Path

from numpy import argmax as np_argmax
from tensorflow import keras

TF_LOG = 'tf_log/'


class ClassifierInterface(ABC):

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError


class TFBaseNet(ClassifierInterface):

    def __init__(self, input_shape, output_shape):
        self._input_shape = input_shape
        self._output_shape = output_shape

        input_tensor, outputs = self._build_graph()
        self._create_model(input_tensor, outputs)

    def _create_model(self, input_tensor, outputs):
        self._model = keras.Model(inputs=input_tensor, outputs=outputs)
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

    def _build_graph(self):
        raise NotImplementedError("Model build function is not implemented")

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y=None, *, validation_data=None, batch_size=None, epochs=1, shuffle=False):
        callbacks = [
            # TensorBoard(
            #     log_dir=TF_LOG + '/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            #     update_freq=1
            #     ),
            # EarlyStopping(monitor='loss', patience=3),
        ]

        best_model_cp_file = Path(TF_LOG).joinpath('models')
        if validation_data is not None:
            best_model_cp_file.mkdir(parents=True, exist_ok=True)
            best_model_cp_file = best_model_cp_file.joinpath('best_model.h5')
            best_model_cp_file.unlink(missing_ok=True)

            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    str(best_model_cp_file),
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True
                )
            )

        self._model.fit(
            x, y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=shuffle
        )

        if validation_data is not None:
            self._model.load_weights(best_model_cp_file)

    def summary(self):
        self._model.summary()

    def evaluate(self, x, y=None):
        self._model.evaluate(x, y)

    def write_summary_to_file(self, filename):
        dirname = Path(__file__).parent
        filename = dirname.joinpath(f'{filename}.txt')
        with open(filename, 'w') as fh:
            self._model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def save(self, filename):
        self._model.save(filename)

    # def load_model(self, filename):
    #     self._model = keras.models.load_model(filename)
