from __future__ import absolute_import, division, unicode_literals

import os
import pickle


class ListPickle():
    """
    """

    pickle_path = os.path.dirname(os.path.abspath(__file__)) + '/default.pickle'
    datalist = []

    def __init__(self, path):
        self.pickle_path = path
        dir_path = os.path.dirname(path)

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        if os.path.isfile(self.pickle_path):
            with open(self.pickle_path, 'rb+') as handle:
                readed = handle.read(4)
                if len(readed) == 0:
                    handle.close()
                    self.save()
                else:
                    handle.seek(0,0)
                    self.datalist = pickle.load(handle)
        else:
            open(self.pickle_path, 'wb+').close()
            self.save()
        

    def save(self, datalist = None):
        if type(datalist) is list:
            self.datalist = datalist
        else:
            self.datalist = [datalist]
        
        with open(self.pickle_path, 'wb') as handle:
            pickle.dump(self.datalist, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_list(self):
        return self.datalist