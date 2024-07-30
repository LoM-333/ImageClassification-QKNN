import os

# to be run once, changing all filenames in test_images_full to [dir_name]_[#].jpg

print(os.getcwd())
categories = [x for x in os.listdir(os.path.join(os.getcwd(), "tests", "test_images_full"))]

for category in categories:
    path = os.path.join(os.getcwd(), "tests", "test_images_full", category)
    for num, img in enumerate(os.listdir(path)):
        nstr = "_" + str(num)
        
        nstr = category + nstr + ".jpg"
        os.rename(os.path.join(path, img), os.path.join(path, nstr))

