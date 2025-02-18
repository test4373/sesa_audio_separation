import contextlib

@contextlib.contextmanager
def open_file(path, mode):
    with open(path, mode) as f:
        yield f

with open_file(output_path, 'wb') as f:
    sf.write(f, estimates.T, sr, subtype=subtype)