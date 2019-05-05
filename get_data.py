import os


def get_data():
    images = []
    folders = os.listdir("images")
    for f in folders:
        images += [(f,x) for x in os.listdir(os.path.join("images",f))]
    return images
