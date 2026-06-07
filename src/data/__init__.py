from data.io import discover_sequences, parse_sequence_filename

__all__ = ["ADFWindowDataset", "discover_sequences", "parse_sequence_filename"]


def __getattr__(name: str):
    if name == "ADFWindowDataset":
        from data.dataset import ADFWindowDataset

        return ADFWindowDataset
    raise AttributeError(name)
