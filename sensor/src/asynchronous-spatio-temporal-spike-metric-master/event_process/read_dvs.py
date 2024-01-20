"""
Function: Read MNIST_DVS data.
Author information: Jianing Li, lijianing@pku.edu.cn, Peking University, Apr 16th, 2018
code based: http://www2.imse-cnm.csic.es/caviar/MNISTDVS.html

"""

import numpy as np
import math
import codecs
import scipy.io

class aefile(object):
    def __init__(self, filename, max_events=1e6):
        self.filename = filename
        self.max_events = max_events
        self.header = []
        self.data, self.timestamp = self.read()

    # alias for read
    def load(self):
        return self.read()

    def read(self):
        with open(self.filename, 'rb') as f:
            line = f.readline()
            line = str(line, encoding = 'gbk')  # bytes to str
            headline = line[0]

            while headline == '#':
                self.header.append(line)
                if line[0:9] == '#!AER-DAT':
                    global aer_version
                    aer_version = line[9]
                current = f.tell()
                line = f.readline()
                headline = chr(line[0])
                # if aer_version != '2':
                #   raise Exception('Invalid AER version. Expected 2, got %s' % aer_version)

            f.seek(0, 2)
            global numEvents
            numEvents = math.floor((f.tell() - current) / 8)

            if numEvents > self.max_events:
                print('There are %i events, but max_events is set to %i. Will only use %i events.' % (
                numEvents, self.max_events, self.max_events))
                numEvents = self.max_events

            f.seek(current, 0)
            timestamps = np.zeros(numEvents)
            data = np.zeros(numEvents)

            for i in range(int(numEvents)):
                hexlify = codecs.getencoder('hex')
                data[i] = int(hexlify(f.read(4))[0], 16)
                timestamps[i] = int(hexlify(f.read(4))[0], 16)

            return data, timestamps

    def save(self, data=None, filename=None, ext='aedat'):
        if filename is None:
            filename = self.filename
        if data is None:
            data = aedata(self)
        if ext is 'aedat':
            # unpack events data
            ts = data.ts
            data = data.pack()

            with open(filename, 'wb') as f:
                # save the head file
                for item in self.header:
                    if type(item) == str:
                        item = bytes(item, encoding='utf-8') # str to bytes
                    f.write(item)

                # save events data
                no_items = len(data)
                for i in range(no_items):
                    f.write(bytes.fromhex(hex(int(data[i]))[2:].zfill(8)))
                    f.write(bytes.fromhex(hex(int(ts[i]))[2:].zfill(8)))

    def unpack(self):
        noData = len(self.data)

        x = np.zeros(noData)
        y = np.zeros(noData)
        t = np.zeros(noData)

        for i in range(noData):
            d = int(self.data[i])

            t[i] = d & 0x1
            x[i] = 128-((d >> 0x1) & 0x7F)
            y[i] = (d >> 0x8) & 0x7F
        return x,y,t


class aedata(object):
    def __init__(self, ae_file=None):
        self.dimensions = (128, 128)
        if isinstance(ae_file, aefile):
            self.x, self.y, self.t = ae_file.unpack()
            self.ts = ae_file.timestamp
        elif isinstance(ae_file, aedata):
            self.x, self.y, self.t = aedata.x, aedata.y, aedata.t
            self.ts = ae_file.ts
        else:
            self.x, self.y, self.t, self.ts = np.array([]), np.array([]), np.array([]), np.array([])

    def __getitem__(self, item):
        rtn = aedata()
        rtn.x = self.x[item]
        rtn.y = self.y[item]
        rtn.t = self.t[item]
        rtn.ts = self.ts[item]
        return rtn

    def __setitem__(self, key, value):
        self.x[key] = value.x
        self.y[key] = value.y
        self.t[key] = value.t
        self.ts[key] = value.ts

    def __delitem__(self, key):
        self.x = np.delete(self.x, key)
        self.y = np.delete(self.y, key)
        self.t = np.delete(self.t, key)
        self.ts = np.delete(self.ts, key)

    def save_to_mat(self, filename):
        scipy.io.savemat(filename, {'X': self.x, 'Y': self.y, 't': self.t, 'ts': self.ts})

    def pack(self):
        noData = len(self.x)
        packed = np.zeros(noData)
        for i in range(noData):
            packed[i] = (int(self.t[i]) & 0x1)
            packed[i] += (int(128 - self.x[i]) & 0x7F) << 0x1
            packed[i] += (int(self.y[i]) & 0x7F) << 0x8

        return packed