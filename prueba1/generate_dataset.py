import os

dir = '/home/uc3m1/PycharmProjects/siameseFaceNet/data/datasets/dataset_prueba'
print(os.listdir(dir))

index_class = 0

with open('./prueba1/test.txt', 'w') as out:
    for person in os.listdir(dir):
        path = os.path.join(dir, person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            out.write(img_path + ' ' + str(index_class) + '\n')
            print(img_path, index_class)

        index_class += 1
