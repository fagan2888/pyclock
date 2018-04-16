from urllib import urlopen
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import (ScalarFormatter)
from matplotlib.dates import (DateFormatter,AutoDateFormatter,AutoDateLocator,seconds,SECONDLY)
from datetime import (date, datetime, time, timedelta)
from operator import itemgetter

class DriftHistory(object):
    def __init__(self, filename):
        self.history = []
        NCOLS = 3
        converters = [lambda x:datetime.fromtimestamp(float(x)),
                      float,
                      lambda x: x.strip()]
        with open(filename, 'rt') as f:
            lines = [_.split(' ',NCOLS-1) for _ in f.readlines()]
        lines = zip(*[map(conv, map(itemgetter(i), lines))
                      for (i,conv) in enumerate(converters)])
        self.history = lines

    def get(self, t0=None, t1=None):
        if t0 is None:
            t0 = datetime.min
        if t1 is None:
            t1 = datetime.max
        return [(dt,adj,comment) for (dt,adj,comment) in self.history
                if (dt >= t0) and (dt <= t1)]


class DataSet(object):
    datafiles = {
        'clock':   {'names': ['drift','amplitude'],
                    'units': ['s','mrad']},
        'weather': {'names': ['temperature','humidity','pressure'],
                    'units': ['deg','.','kPa']},
    }

    @staticmethod
    def get_datafile_for_channel(channel):
        for datafile,info in DataSet.datafiles.items():
            if channel in info['names']:
                return datafile
        raise ValueError('Unknown channel "%s"' % channel)

    @staticmethod
    def get_channels_in_datafile(datafile):
        return DataSet.datafiles[datafile]['names']

    @staticmethod
    def get_units(channel):
        for datafile,info in DataSet.datafiles.items():
            for name,unit in zip(info['names'],info['units']):
                if channel == name: return unit
        raise ValueError('Unknown channel "%s"' % channel)

    def __init__(self, dataurl):
        self.dataurl = dataurl

    def get_day_url(self, day, datafile='clock'):
        return self.dataurl \
               + day.strftime('/%Y/%m/{0}%Y-%m-%d.txt'.format(datafile))

    def get_data(self, t0=None, t1=None, datafile='clock'):
        if t0 is None:
            t0 = datetime.combine(date.today()-timedelta(days=7), time(0,0,0))
        if t1 is None:
            t1 = datetime.combine(date.today()-timedelta(days=7), time(23,59,59))

        t = t0
        data = []
        conv = {0: lambda x:datetime.fromtimestamp(float(x))}
        while t <= t1:
            f = urlopen(self.get_day_url(t, datafile=datafile))
            d = np.genfromtxt(f, dtype=None, converters=conv,
                              names=['time']+DataSet.datafiles[datafile]['names'])
            f.close()
            data.append(d)
            t += timedelta(days=1)
        data = np.concatenate(data)
        data = data[np.logical_and(data['time'] >= t0, data['time'] <= t1)]
        return data


class SensibleDateFormatter(AutoDateFormatter):
    def __init__(self, locator, tz=None):
        """
        Choose format prefix based on data range.
        Choose whether to include seconds based on unit.
        """
        AutoDateFormatter.__init__(self, locator, tz)
        self.scaled = {
            1./(24*60*60): '%H:%M:%S',
            1.           : '%H:%M',
        }
        self.prefixd = {
            365. : '%Y %b %d ',
            1.   : '%b %d ',
        }

    def __call__(self, x, pos=0):
        scale = float( self._locator._get_unit() )
        dmin,dmax = self._locator.axis.get_view_interval()

        fmt = ''
        for k in sorted(self.prefixd):
            if k < (dmax-dmin):
                fmt = self.prefixd[k]
                break

        for k in sorted(self.scaled):
           if k>=scale:
              fmt += self.scaled[k]
              break

        self._formatter = DateFormatter(fmt, self._tz)
        return self._formatter(x, pos)


class ClockPlot(object):
    def __init__(self, dataset, drifthistory=None):
        self.dataset = dataset
        self.comments = drifthistory

    def plot(self, channel, t0=None, t1=None):
        derived_channels = {'air density': ('weather','kg/m3')}
        datafile = (channel in derived_channels) and derived_channels[channel][0] \
            or DataSet.get_datafile_for_channel(channel)
        data = self.dataset.get_data(t0, t1, datafile)

        if channel == 'air density':
            y = self.calc_airdensity(data)
        else:
            y = data[channel]

        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(data['time'], y,'.-')

        locator = AutoDateLocator(minticks=8,interval_multiples=True)
        locator.intervald[SECONDLY] = [3,6,12,15,30]
        ax.xaxis.set_major_locator( locator )
        formatter = SensibleDateFormatter(locator)
        ax.xaxis.set_major_formatter( formatter )

        ax.locator_params(axis='y', prune='lower')
        yformatter = ScalarFormatter(useOffset=False)
        ax.yaxis.set_major_formatter(yformatter)
        ax.grid()

        units = (channel in derived_channels) and derived_channels[channel][1] \
            or DataSet.get_units(channel)
        yformatter.set_locs(ax.yaxis.get_major_locator()()) # force update of yformatter
        ax.set_title('%s (mean = %s %s)' % (channel.title(), yformatter(y.mean()), units))
        ax.set_ylabel(units)
        fig.autofmt_xdate()
        ax.autoscale(False)

        # symbols for comments
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        for t, drifthist, comment in self.comments.get(data['time'][0],data['time'][-1]):
            m = re.match(r'ADJUST[^-+.0-9]*([-+.0-9]+)', comment)
            adj = m and float(m.group(1)) or 0
            if   adj > 0: c,s = 'red', '^'
            elif adj < 0: c,s = 'blue','v'
            else:         c,s = 'black','o'
            ax.plot(t, 0.98, s, color=c, transform=trans)
            ax.axvline(t, color=c, alpha=0.5)

    def calc_airdensity(self, data):
        # Constants for density calculation
        Rd = 287.05
        Rv = 461.495

        Tk = data['temperature'] + 273.15
        Psat = 610.78 * 10**((7.5*Tk-2048.625)/(Tk-35.85))
        Pv = (data['humidity']/100)*Psat
        rhod = (100*data['pressure']-Pv) / (Rd*Tk)
        rhov = Pv / (Rv*Tk)
        return (rhov + rhod)


if __name__ == '__main__':
    ds = DataSet('sample-data')
    #data = ds.get_data(t0=datetime(2010,12,18), t1=datetime(2010,12,20,23,59,59), datafile='clock')

    dh = DriftHistory('sample-data/drift-history.txt')

    cp = ClockPlot(ds,dh)
    cp.plot('air density', t0=datetime(2010,12,15), t1=datetime(2010,12,18,23,59,59))
    plt.show()
