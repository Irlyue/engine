from datetime import datetime


class Logger:
    def print(self, s):
        raise NotImplementedError

    def close(self):
        pass


class TerminalFileLogger(Logger):
    def __init__(self, log_file='/tmp/log.txt'):
        self.log_file = open(log_file, 'a')
        self.print('==> Logging start at %s' % datetime.now())

    def print(self, s):
        print(s)
        print(s, file=self.log_file, flush=True)

    def close(self):
        self.log_file.close()
