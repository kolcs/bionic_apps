"""
Online plotter for EEG

@license: PPKE ITK, MTA TTK
@author: Köllőd Csaba, kollod.csaba@ikt.ppke.hu
"""
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg


class EEGPlot:
    """
    Helper class for online EEG plotter.
    Each EEG channel has it's own plot.
    This class keeps track about it's own plot, which can be updated
    """

    def __init__(self, parent, win, index, title=''):
        """

        :param parent: EEGPlotter class object
        :param win: plot window
        :param index: index of data which sould be plotted from the data list
        :param title: title for the plot, could be a channel name
        """
        self.index = index
        self.parent = parent
        p = win.addPlot(title=title)
        p.setRange(yRange=[-0.0002, 0.0002])
        self.curve = p.plot(pen='y')  # define color here

    def update(self, ev=None):
        """
        update function, which is called every n iteration
        :param ev: If Qt requires it...
        """
        data = self.get_data()
        if data is not None:
            self.curve.setData(data)

    def get_data(self):
        """
        called in update function
        :return: returns the required data which should be plotted
        """
        data = self.parent._get_data(self.index)
        return data


class EEGPlotter:

    def __init__(self, wx=1000, wy=800, plot_size=(6, 2), title="Online BCI plotter",
                 win_title='EEG filter plotting'):
        """
        Online EEG plotter
        :param wx: window width
        :param wy: window height
        :param plot_size: tuple to organise the plots. max 14 is suggested
        :param title: title of program
        :param win_title: title of window
        """
        self._app = QtGui.QApplication([])
        self._win = pg.GraphicsWindow(title=title)
        self._win.resize(wx, wy)
        self._win.setWindowTitle(win_title)
        pg.setConfigOptions(antialias=True)
        self._timer = QtCore.QTimer()
        self._curves = list()
        self._data_source = None
        self._data = list()
        self._plots = list()
        self._plot_size = plot_size
        self._build_graphs()

    def _build_graphs(self):
        """
        builds up the required plots
        """

        if type(self._plot_size == 'tuple'):
            cols = self._plot_size[1]
            rows = self._plot_size[0]
        else:
            rows = self._plot_size
            cols = 1
        index = 0
        for i in range(rows):
            for j in range(cols):
                self._add_plot(index)
                index += 1
            self._next_row()

    def _add_time_ev_func(self, func):
        self._timer.timeout.connect(func)

    def _add_plot(self, index, title=''):
        p = EEGPlot(self, self._win, index, title)
        self._plots.append(p)
        self._add_time_ev_func(p.update)

    def add_data_source(self, source):
        """
        define data source here
        :param source: data source
        """
        self._data_source = source

    def _next_row(self):
        self._win.nextRow()

    def _get_data(self, index):
        if len(self._data) > 0:
            return self._data[index]
        else:
            return None

    def _update_data(self, ev=None):
        self._data = self._data_source.get_data()

    def run(self):
        """
        Run GUI
        """
        self._add_time_ev_func(self._update_data)
        self._timer.start(20)
        QtGui.QApplication.instance().exec_()


class Source:
    """
    Helper class for testing...
    """

    def get_data(self):
        return [np.random.normal(size=160) for _ in range(2 * 24)]


if __name__ == '__main__':
    plotter = EEGPlotter()
    plotter.add_data_source(Source())
    plotter.run()
