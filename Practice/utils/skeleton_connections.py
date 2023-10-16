# NOTE: This file is not used in the project. It is only used for reference.
# These connections are used to draw the skeleton on the image.
# Be careful to use the correct skeleton shape that is composed by 32 joints.
# See test loop in order to have a guide on how create 32 joints from 22 joints.

connect = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (0, 6),
            (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (24, 25), (24, 17),
            (24, 14), (14, 15)
    ]
LR = [
        False, True, True, True,
        True, True, False, False,
            False, False,
        False, True, True, True, True, True, True, 
        False, False, False, False, False, False, False, True,
        False, True, True, True, True,
        True, True
]