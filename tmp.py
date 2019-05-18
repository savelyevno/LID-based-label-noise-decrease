import os

folder_path = '/Users/nikita/Desktop/untitled folder/prefabs'

files = os.listdir(folder_path)
files2 = []
for it in files:
    if it != '.DS_Store':
        files2.append(it)

Id = 400
for filename in sorted(files2, key=lambda s: int(s.split(' ')[0])):
    filename_split = filename.split(' ')
    # Id = int(filename_split[0])
    #
    # new_Id = Id
    # if 339 <= Id <= 347:
    #     new_Id = Id - 6
    # elif 351 <= Id <= 382:
    #     new_Id = Id - (6 + 3)
    #
    # if new_Id != Id:
    #     new_filename = str(new_Id) + ' ' + str.join(' ', filename_split[1:])
    #     os.rename(os.path.join(folder_path, filename),
    #               os.path.join(folder_path, new_filename))

    new_filename = str(Id) + ' ' + str.join(' ', filename_split[1:])
    os.rename(os.path.join(folder_path, filename),
              os.path.join(folder_path, new_filename))

    Id += 1
