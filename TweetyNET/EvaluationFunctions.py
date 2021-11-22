def frame_error(pred, actual):
    if len(pred) != len(actual):
        print("Incorrect Lengths: ", len(pred), len(actual))
        return 0
    match_up = 0
    for i in range(len(pred)):
        if pred[i] == actual[i]:
            match_up += 1
    return match_up  # /len(pred)


# This only matters if things are in the correct order. Can I split by video file?

def syllable_edit_distance(pred, actual):
    if len(pred) != len(actual):
        print("Incorrect Lengths: ", len(pred), len(actual))
        return 0
    distances = range(len(pred) + 1)
    for i2, c2 in enumerate(actual):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(pred):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]  # /len(pred)
