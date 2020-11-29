import json


def read_json(fn):
    with open(fn, 'rb') as f:
        data = json.load(f)
    return data


def write_json(data, fn):
    with open(fn, 'wb') as f:
        json.dump(data, f, indent = 2)


def resize_image_wrt_asp(img, factor):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    new_img = img.resize((new_w, new_h))
    return new_img