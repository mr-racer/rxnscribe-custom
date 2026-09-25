SHIM = True


class Reader:  # benchmark stand-in: no OCR
    def __init__(self, *args, **kwargs):
        pass

    def readtext(self, image, detail=0, **kwargs):
        return []
