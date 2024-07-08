import errno
import os

class OutStream:
    """
    Solution to get output from singularity in real time from:
    https://tbrink.science/blog/2017/04/30/processing-the-output-of-a-subprocess-with-python-in-realtime/
    """
    def __init__(self, fileno):
        self._fileno = fileno
        self._buffer = b""

    def read_lines(self):
        try:
            output = os.read(self._fileno, 1000)
        except OSError as e:
            if e.errno != errno.EIO: raise
            output = b""
        lines = output.split(b"\n")
        lines[0] = self._buffer + lines[0] # prepend previous
                                           # non-finished line.
        if output:
            self._buffer = lines[-1]
            finished_lines = lines[:-1]
            readable = True
        else:
            self._buffer = b""
            if len(lines) == 1 and not lines[0]:
                # We did not have buffer left, so no output at all.
                lines = []
            finished_lines = lines
            readable = False
        finished_lines = [line.rstrip(b"\r").decode()
                          for line in finished_lines]
        return finished_lines, readable

    def fileno(self):
        return self._fileno

