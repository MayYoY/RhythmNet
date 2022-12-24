class FrameTrain:
    """for RhythmNet, dynamic detection and no large"""
    input_path = ""
    cache_path = "./cache"
    record_path = "./record.csv"

    MODIFY = True
    W = -1
    H = -1
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = 1
    CROP_FACE = True
    LARGE_FACE_BOX = True
    LARGE_BOX_COEF = 1.2

    DO_CHUNK = True
    CHUNK_LENGTH = 300
    CHUNK_STRIDE = -1


class FrameTest:
    """for RhythmNet, dynamic detection and no large"""
    input_path = ""
    cache_path = "./cache"
    record_path = "./test_record.csv"

    MODIFY = False
    W = 120
    H = 120
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = 1
    CROP_FACE = True
    LARGE_FACE_BOX = True
    LARGE_BOX_COEF = 1.2

    DO_CHUNK = True
    CHUNK_LENGTH = 300
    CHUNK_STRIDE = -1
