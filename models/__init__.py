from torch import nn

from models.CKAN import CKAN

__all__ = [
    'CKAN', 'Parallel'
]


class Parallel(nn.DataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)

