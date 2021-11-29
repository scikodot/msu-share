import os

def trace(speed_limit, violation_threshold):
    """Trace one object's violation time"""
    violation_time = 0
    t_prev, x_prev, y_prev = 0, 0, 0
    while True:
        t, x, y = yield

        if t_prev != 0:
            dt = t - t_prev
            dx = x - x_prev
            dy = y - y_prev

            distance = (dx**2 + dy**2)**0.5
            speed = distance / dt * 3.6

            if speed > speed_limit:
                violation_time += dt
            else:
                violation_time = 0

        yield violation_time >= violation_threshold
        t_prev, x_prev, y_prev = t, x, y

def violators(file):
    """Find objects that are violators"""
    traces = {}
    speed_limit, violation_threshold = 40, 1
    with open(file, 'r', newline='', encoding='utf8') as csvfile:
        next(csvfile)
        for line in csvfile:
            # Read and parse line
            vals = line.split(',')
            i = vals[1]
            t, x, y = float(vals[0]), float(vals[3]), float(vals[4])

            # Add new trace
            if i not in traces:
                tr = trace(speed_limit, violation_threshold)
                next(tr)
                traces[i] = tr

            tr = traces[i]

            # Skip violators
            if not tr:
                continue

            violator = tr.send((t, x, y))
            next(tr)

            if violator:
                # Invalidate trace
                traces[i] = False
                yield i

if __name__ == '__main__':
    dir_ = os.path.dirname(__file__)
    with open(os.path.join(dir_, "output/answer.txt"), 'w', encoding='utf8') as output:
        for v in violators(os.path.join(dir_, "input/data.csv")):
            output.write(str(v) + '\n')
