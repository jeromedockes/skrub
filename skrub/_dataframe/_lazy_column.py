from dataclasses import dataclass
from typing import Any


@dataclass
class PolarsLazyColumn:
    name: str
    dataframe: Any

    @property
    def dtype(self):
        return self.dataframe.schema[self.name]
