import os


subfolders = [f.path for f in os.scandir(os.getcwd()) if f.is_dir() ] 
subfolders.remove(os.getcwd() + '\\face_recognition')
for item in subfolders:
    print(item)   


subfolders = [f.path for f in os.scandir(os.getcwd()) if f.is_dir() ] 
subfolders.remove(os.getcwd() + r'\face_recognition')
for item in subfolders:
    obj_names.append(item)
    for file_name in os.listdir(item):
        obj_images.append(face_recognition.load_image_file(item + '\\' +file_name))

        
for x in obj_images:
    print(x)
    obj_encs.append(face_recognition.face_encodings(x))