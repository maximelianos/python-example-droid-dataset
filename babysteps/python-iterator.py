def my_iterator():
    # yields 0, 1, and stops
    for i in range(5):
        if i == 2:
            return
        yield i

ints = [x for x in my_iterator()]
print(ints)
