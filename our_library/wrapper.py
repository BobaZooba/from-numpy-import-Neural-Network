from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from our_library.tools import binary_accuracy, multi_class_f1


class Wrapper:
    def __init__(
        self,
        model,
        loss,
        learning_rate=0.001,
        multi_class=False,
        custom_metric_function=None,
        verbose=True,
    ):

        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate

        self.multi_class = multi_class

        self.metric_function = custom_metric_function or (
            binary_accuracy if not self.multi_class else multi_class_f1
        )

        self.verbose = verbose

        self.losses = []
        self.losses_per_epoch = []

        self.metric = []
        self.metric_per_epoch = []
        self.metric_per_epoch_test = []

    def train(self, train_loader, test_loader, epochs=15):

        progress_bar = None

        try:

            for n_epoch in range(epochs):

                epoch_losses = []
                epoch_metric = []
                epoch_metric_test = []

                progress_bar = tqdm(
                    total=len(train_loader),
                    desc="Epoch {}".format(n_epoch + 1), disable=not self.verbose
                )

                for batch in train_loader:

                    x, y = self.batch_processing(batch)

                    loss_batch, prediction = self.train_batch(x, y)

                    self.losses.append(loss_batch)
                    epoch_losses.append(loss_batch)

                    batch_metric = self.metric_function(y, prediction)

                    self.metric.append(batch_metric)
                    epoch_metric.append(batch_metric)

                    progress_bar.update()

                    progress_bar.set_postfix(loss=np.mean(epoch_losses), metric=np.mean(epoch_metric))

                for batch in test_loader:

                    x, y = self.batch_processing(batch)

                    prediction = self.predict_batch(x)

                    batch_accuracy = self.metric_function(y, prediction)

                    epoch_metric_test.append(np.mean(batch_accuracy))

                self.losses_per_epoch.append(np.mean(epoch_losses))
                self.metric_per_epoch.append(np.mean(epoch_metric))
                self.metric_per_epoch_test.append(np.mean(epoch_metric_test))

                progress_bar.close()

        except KeyboardInterrupt:

            if progress_bar:
                progress_bar.close()

    @staticmethod
    def batch_processing(batch):

        x, y = batch

        x = x.view(x.shape[0], -1).numpy()
        y = y.numpy()

        return x, y

    def train_batch(self, x, y):
        raise NotImplementedError

    def predict_batch(self, x):
        return self.model.forward(x)

    def plot(self):

        plt.figure(figsize=(16, 12))

        plt.subplot(1, 2, 1)

        plt.grid()
        plt.xlabel("Training step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.plot(self.losses)

        plt.subplot(1, 2, 2)

        plt.grid()
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Metric", fontsize=12)
        plt.plot(self.metric_per_epoch, label="Train")
        plt.plot(self.metric_per_epoch_test, label="Valid")
        plt.ylim(0, 1)
        plt.legend()

        plt.show()


class MNISTWrapper(Wrapper):
    def __init__(self, model, loss, learning_rate=0.001, multi_class=False):

        # передаем параметры, чтобы отработал __init__ от наследуемого класса
        # можно было бы сделать через **kwargs, но сделал так для наглядности и чтобы были подсказки в юпитере

        super().__init__(
            model=model, loss=loss, learning_rate=learning_rate, multi_class=multi_class
        )

    def train_batch(self, x, y):
        """
        Нужно реализовать одну итерацию обучения модели:
        1. Рассчет forward
        2. Рассчет функции потерь
        3. Рассчет backward от функции потерь
        4. Рассчет backward по модели
        5. Обновление весов
        :param x: входные данные np.array with shape (batch_size, n_features)
        :param y: предсказания np.array with shape (batch_size, n_classes)
        :return:
        loss_batch - значение функции потерь, просто скаляр
        prediction - матрица предсказаний вашей модели

        напомню важные штуки, которые знает наш класс:
        self.model
        self.loss
        self.learning_rate
        """
        ...

    def predict_batch(self, x):
        """
        Предсказание (aka inference) вашей модели:
        1. Рассчет forward
        :param x: входные данные np.array with shape (batch_size, n_features)
        :return: prediction - матрица предсказаний вашей модели
        """
        ...
