import datetime
import signal


class Timer:
    def __init__(self):
        self.time_stack = []

    def start(self):
        self.time_stack.append(datetime.datetime.now())

    def stop(self):
        then = self.time_stack.pop()
        now = datetime.datetime.now()
        return now - then

    def stop_start(self):
        res = self.stop()
        self.start()

        return res


class Ticker:
    def __init__(self):
        self.handler_id = 0
        self.handlers = []

    def start_track(self, period, action):
        this_handler_id = self.handler_id

        def handler(signum, frame):
            action()
            if self.handlers[this_handler_id] is not None:
                signal.alarm(period)

        self.handlers.append(handler)

        signal.signal(signal.SIGALRM, self.handlers[self.handler_id])
        signal.alarm(period)

        self.handler_id += 1

        return self.handler_id - 1

    def stop_track(self, handler_id):
        self.handlers[handler_id] = None


timer = Timer()
ticker = Ticker()

