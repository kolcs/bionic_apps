from pathlib import Path

import pandas as pd


class ResultHandler:
    def __init__(self, fix_params, changing_params, to_beginning=(), filename=None):
        self._fix = fix_params.copy()
        self._changing = changing_params
        self.res = pd.DataFrame(columns=list(fix_params) + changing_params)
        self.filename = filename

        # reorder columns:
        cols = list(self.res.columns)
        for c in to_beginning:
            cols.remove(c)
        self.res = self.res[list(to_beginning) + cols]

    def add(self, res_dict):
        assert len(self._changing) == len(res_dict), f'Expected {len(self._changing)} results. ' \
                                                     f'Got {len(res_dict)} instead.'
        assert all(k in self._changing for k in res_dict), 'Name of parameters are not match with ' \
                                                           'changing_params.'
        res_len = len(pd.DataFrame(res_dict))
        if res_len == 1:
            res_dict.update(self._fix)
        else:
            fix = {key: [val] * res_len for key, val in self._fix.items()}
            res_dict.update(fix)
        res_dict = pd.DataFrame(res_dict)
        self.res = self.res.append(res_dict, ignore_index=True)

    def save(self, filename=None, sep=';', encoding='utf-8', index=False):
        assert filename is not None or self.filename is not None, 'filename must be defined!'
        if isinstance(filename, str):
            self.filename = filename
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        self.res.to_csv(self.filename, sep=sep, encoding=encoding, index=index)

    def __len__(self):
        return len(self.res)
