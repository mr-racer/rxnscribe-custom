class COCO:  # benchmark stand-in, never used at inference
    def __init__(self, *args, **kwargs):
        raise RuntimeError("pycocotools is not installed (benchmark shim)")
