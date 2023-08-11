import csv, re, os, sys
from datetime import datetime

csv.field_size_limit(min(sys.maxsize, 0x800000))

class CsvParser():
    """
        get_row_list (column=[], limit=0)
        return List
    """
    file = None
    file_content = None
    is_multiple = False
    file_extension = re.compile("^(.*).csv?$", re.IGNORECASE)
    columns = []
    column_map = {}
    data = []

    def __init__(self, **kwargs):

        self.file = kwargs.get(
            'file',
            None,
        )

        if self.file:
            start_time = datetime.now()

            if os.path.isdir(self.file):

                self.is_multiple = True

                for _file in os.listdir(self.file):

                    if self.file_extension.search(_file):

                        # _file_path = '{}/{}'.format(self.file, _file)
                        # with open(_file_path) as csvfile:
                        #     spamer = csv.reader(csvfile, delimiter=',', quotechar='|')
                        #     self.file_content = spamer
                        pass

            else:

                self.is_multiple = False
                with open(self.file, encoding='utf-8', newline='') as csvfile:
                    spamer = csv.reader(csvfile, delimiter=',', quotechar='"')
                    columns = []
                    datas = []
                    self.file_content = spamer
                    _i = 0
                    for row in spamer:
                        row = [re.sub(r'[\t\r\n\'\"]', '', r) for r in row]
                        if _i == 0:
                            columns = row
                        else:
                            datas.append(row)
                        _i += 1

                    column_map = {}
                    for idx, col in enumerate(columns):
                        column_map[col] = idx
                    self.column_map = column_map
                    self.columns = columns
                    self.data = datas


            end_time = datetime.now()
            spend_second = (end_time - start_time).total_seconds()

            print('====CsvParser Loads File spend seconds: ', spend_second)
            # print('self.data[:10]: ', self.data[:10])

        else:

            print('CsvParser Failed With Wrong File path.')


    def get_row_list(self, column=[], limit=0):
        scm = self.column_map
        total_rows = []
        _indices = []
        
        for col in column:
            __idx = -1
            if type(col) == list:
                for __c in col:
                    __loc_idx = scm.get(__c, -1)
                    if __loc_idx >= 0:
                        __idx = __loc_idx
                        break
            else:
                __idx = scm.get(col, -1)

            _indices.append(__idx)
        
        # print('_indices: ', _indices)
        # print('self.data', self.data[-1:])
        _i = limit
        if limit <= 0:
            _i = len(self.data) -1
        for d in self.data:
            row = []
            for ic in _indices:
                row.append(d[ic])
            total_rows.append(row)
            _i -= 1
            if _i < 0:
                break
            
            
        return total_rows
    