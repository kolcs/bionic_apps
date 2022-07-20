from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy import argmax as np_argmax
from tensorflow import keras
from tensorflow.python.platform import tf_logging as logging


class ClassifierInterface(ABC):

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError


class EarlyStoppingAfterInitValidationReached(keras.callbacks.EarlyStopping):

    def __init__(self, monitor='val_loss', give_up=100, min_delta=0, patience=0,
                 verbose=0, mode='auto', baseline=None, restore_best_weights=False):
        super(EarlyStoppingAfterInitValidationReached, self).__init__(monitor, min_delta, patience, verbose, mode,
                                                                      baseline, restore_best_weights)
        self.init_value = None
        self.reached_init = False
        self.give_up = give_up
        self.g_ind = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if epoch == 0:
            self.init_value = current
        elif not self.reached_init:
            if self._is_improvement(current, self.init_value):
                self.reached_init = True
            else:
                self.g_ind += 1
                self.wait = 0

        # Only check after the first epoch.
        if (self.wait >= self.patience and epoch > 0) or self.g_ind > self.give_up:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    keras.utis.io_utils.print_msg(
                        'Restoring model weights from the end of the best epoch: '
                        f'{self.best_epoch + 1}.')
                self.model.set_weights(self.best_weights)


class SaveAndRestoreBestModel(keras.callbacks.Callback):

    def __init__(self):
        super(SaveAndRestoreBestModel, self).__init__()
        self._val_loss = 'val_loss'
        self._val_acc = 'val_accuracy'
        self._min_loss = np.Inf
        self._max_acc = -np.Inf
        self.best_weights = None
        self.epoch = 0
        self._init_loss = 0
        self._reached_init = False

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self._min_loss = np.Inf
        self._max_acc = -np.Inf
        self.best_weights = None
        self._init_loss = 0
        self._reached_init = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = self._get_monitor_value(self._val_loss, logs)
        val_acc = self._get_monitor_value(self._val_acc, logs)
        if val_loss is None or val_acc is None:
            return

        if epoch == 0:
            self._init_loss = val_loss
        elif val_loss < self._init_loss:
            self._reached_init = True

        if not self._reached_init:
            if val_acc > self._max_acc:
                self.epoch = epoch
                self._max_acc = val_acc
                self.best_weights = self.model.get_weights()

        elif val_loss < self._min_loss and val_acc > self._max_acc:
            self.epoch = epoch
            self._min_loss = val_loss
            self._max_acc = val_acc
            self.best_weights = self.model.get_weights()

    @staticmethod
    def _get_monitor_value(value, logs):
        logs = logs or {}
        monitor_value = logs.get(value)
        if monitor_value is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            value, ','.join(list(logs.keys())))
        return monitor_value

    def on_train_end(self, logs=None):
        print(f'Restored model at epoch{self.epoch + 1} with '
              f'val_loss={self._min_loss:.4f} and val_acc={self._max_acc:.4f}')
        self.model.set_weights(self.best_weights)


def _reset_weights(model):
    # https://github.com/keras-team/keras/issues/341
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # if you're using a model as a layer
            _reset_weights(layer)  # apply function recursively
            continue

        # where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key:  # is this item an initializer?
                continue  # if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer':  # special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            # use the initializer
            if var is not None:
                var.assign(initializer(var.shape, var.dtype))


def _reinitialize_model(model):
    # https://github.com/tensorflow/tensorflow/issues/48230
    weights = []
    initializers = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # if you're using a model as a layer
            _reinitialize_model(layer)  # apply function recursively
            continue
        elif isinstance(layer, keras.layers.DepthwiseConv2D):
            weights += [layer.depthwise_kernel, layer.bias]
            initializers += [layer.depthwise_initializer, layer.bias_initializer]
        elif isinstance(layer, keras.layers.SeparableConv2D):
            weights += [layer.depthwise_kernel, layer.pointwise_kernel, layer.bias]
            initializers += [layer.depthwise_initializer, layer.pointwise_initializer, layer.bias_initializer]
        elif isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
            weights += [layer.kernel, layer.bias]
            initializers += [layer.kernel_initializer, layer.bias_initializer]
        elif isinstance(layer, keras.layers.BatchNormalization):
            weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
            initializers += [layer.gamma_initializer,
                             layer.beta_initializer,
                             layer.moving_mean_initializer,
                             layer.moving_variance_initializer]
        elif isinstance(layer, keras.layers.Embedding):
            weights += [layer.embeddings]
            initializers += [layer.embeddings_initializer]
        elif isinstance(layer, (keras.layers.InputLayer,
                                keras.layers.Reshape,
                                keras.layers.Concatenate,
                                keras.layers.Lambda,
                                keras.layers.Activation,
                                keras.layers.AveragePooling2D,
                                keras.layers.Dropout,
                                keras.layers.Flatten,
                                )):
            # These layers don't need initialization
            continue
        else:
            raise ValueError('Unhandled layer type: %s' % (type(layer)))
    for w, init in zip(weights, initializers):
        if w is not None:
            w.assign(init(w.shape, dtype=w.dtype))


class TFBaseNet(ClassifierInterface):

    def __init__(self, input_shape, output_shape, save_path='tf_log/'):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._save_path = Path(save_path)

        input_tensor, outputs = self._build_graph()
        self._create_model(input_tensor, outputs)

    def _create_model(self, input_tensor, outputs):
        self._model = keras.Model(inputs=input_tensor, outputs=outputs)
        self._model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        _reinitialize_model(self._model)

    def _build_graph(self):
        raise NotImplementedError("Model build function is not implemented")

    def predict(self, x):
        predictions = self._model.predict(x)
        return np_argmax(predictions, axis=-1)

    def fit(self, x, y=None, *, validation_data=None, batch_size=None, epochs=1,
            shuffle=False, patience=15, verbose='auto'):
        callbacks = [
            # TensorBoard(
            #     log_dir=TF_LOG + '/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S"),
            #     update_freq=1
            #     ),
            # EarlyStopping(monitor='loss', patience=3),
        ]

        # best_model_cp_file = self._save_path.joinpath('models')
        if validation_data is not None:
            # best_model_cp_file.mkdir(parents=True, exist_ok=True)
            # best_model_cp_file = best_model_cp_file.joinpath('best_model.h5')
            # best_model_cp_file.unlink(missing_ok=True)

            monitor = 'val_loss'  # 'val_accuracy'
            mode = 'min'  # 'max'

            callbacks.extend((
                SaveAndRestoreBestModel(),
                EarlyStoppingAfterInitValidationReached(
                    monitor=monitor,
                    min_delta=0,
                    patience=patience,
                    mode=mode,
                )
            ))

        self._model.fit(
            x, y,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=shuffle,
            verbose=verbose
        )

        # if validation_data is not None:
        #     self._model.load_weights(best_model_cp_file)

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
