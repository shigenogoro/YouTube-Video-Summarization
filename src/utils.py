def data_generator(dataset):
    """
    Create a generator that yields (id, summary, transcript).
    """
    for instance in dataset:
        yield instance["id"], instance["summary"], instance["transcript"]
